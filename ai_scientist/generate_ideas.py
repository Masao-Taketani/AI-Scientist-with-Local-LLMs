import json
import os
import os.path as osp
import time
from typing import List, Dict, Union
import logging

import backoff
import requests

from ai_scientist.llm import get_response_from_local_llm, extract_json_between_markers, create_client


idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


# GENERATE IDEAS
def generate_ideas(
        base_dir,
        platform,
        client,
        model_or_pipe,
        skip_generation=False,
        max_num_generations=20,
        num_reflections=5,
        show_thought=False,
):
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    for _ in range(max_num_generations):
        print()
        print(f"Generating idea {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)
    
            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            text, msg_history = get_response_from_local_llm(
                idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                platform=platform,
                client=client,
                model_or_pipe=model_or_pipe,
                system_message=idea_system_prompt,
                msg_history=msg_history,
                show_thought=show_thought,
            )
            ## PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)
    
            # Iteratively improve task.
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_local_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        platform=platform,
                        client=client,
                        model_or_pipe=model_or_pipe,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                        show_thought=show_thought,
                    )
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert (
                            json_output is not None
                    ), "Failed to extract JSON from LLM output"
                    print(json_output)
    
                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break
    
            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


# GENERATE IDEAS OPEN-ENDED
def generate_next_idea(
        base_dir,
        platform,
        client,
        model_or_pipe,
        prev_idea_archive=[],
        num_reflections=5,
        max_attempts=10,
        show_thought=False,
):
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"Generating idea {original_archive_size + 1}")

    if len(prev_idea_archive) == 0:
        print(f"First iteration, taking seed ideas")
        # seed the archive on the first run with pre-existing ideas
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas[:1]:
            idea_archive.append(seed_idea)
    else:
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        for _ in range(max_attempts):
            try:
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea))
                prev_ideas_string = "\n\n".join(idea_strings)

                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                text, msg_history = get_response_from_local_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    )
                    + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                    platform=platform,
                    client=client,
                    model_or_pipe=model_or_pipe,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                    show_thought=show_thought,
                )
                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"
                print(json_output)

                # Iteratively improve task.
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        text, msg_history = get_response_from_local_llm(
                            idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            platform=platform,
                            client=client,
                            model_or_pipe=model_or_pipe,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                            show_thought=show_thought,
                        )
                        ## PARSE OUTPUT
                        json_output = extract_json_between_markers(text)
                        assert (
                                json_output is not None
                        ), "Failed to extract JSON from LLM output"
                        print(json_output)

                        if "I am done" in text:
                            print(
                                f"Idea generation converged after {j + 2} iterations."
                            )
                            break

                idea_archive.append(json_output)
                break
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

    ## SAVE IDEAS
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10, engine="openalex") -> Union[None, List[Dict]]:
    if not query:
        return None
    if engine == "semanticscholar":
        S2_API_KEY = os.getenv("S2_API_KEY")
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": S2_API_KEY} if S2_API_KEY else {},
            params={
                "query": query,
                "limit": result_limit,
                "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
            },
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(
            f"Response Content: {rsp.text[:500]}"
        )  # Print the first 500 characters of the response content
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        time.sleep(1.0)
        if not total:
            return None

        papers = results["data"]
        return papers
    elif engine == "openalex":
        import pyalex
        from pyalex import Work, Works
        mail = os.environ.get("OPENALEX_MAIL_ADDRESS", None)
        if mail is None:
            print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!")
        else:
            pyalex.config.email = mail

        def extract_info_from_work(work: Work, max_abstract_length: int = 1000) -> dict[str, str]:
            # "Unknown" is returned when venue is unknown...
            venue = "Unknown"
            for i, location in enumerate(work["locations"]):
                if location["source"] is not None:
                    venue = location["source"]["display_name"]
                    if venue != "":
                        break
            title = work["title"]
            abstract = work["abstract"]
            if abstract is None:
                abstract = ""
            if len(abstract) > max_abstract_length:
                # To avoid context length exceed error.
                print(f"[WARNING] {title=}: {len(abstract)=} is too long! Use first {max_abstract_length} chars.")
                abstract = abstract[:max_abstract_length]
            authors_list = [author["author"]["display_name"] for author in work["authorships"]]
            authors = " and ".join(authors_list) if len(authors_list) < 20 else f"{authors_list[0]} et al."
            paper = dict(
                title=title,
                authors=authors,
                venue=venue,
                year=work["publication_year"],
                abstract=abstract,
                citationCount=work["cited_by_count"],
            )
            return paper

        works: List[Dict] = Works().search(query).get(per_page=result_limit)
        papers: List[Dict[str, str]] = [extract_info_from_work(work) for work in works]
        return papers
    elif engine == "core":
        CORE_API_KEY = os.getenv("CORE_API_KEY")
        headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
        payload = {"q": query, "limit": result_limit}
        rsp = requests.post(
            "https://api.core.ac.uk/v3/search/works", 
            data=json.dumps(payload), 
            headers=headers
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(
            f"Response Content: {rsp.text[:500]}"
        )  # Print the first 500 characters of the response content
        rsp.raise_for_status()
        results = rsp.json()
        
        total = results["totalHits"]
        if total == 0:
            return no_query
        papers = results["results"]
        return papers
    else:
        raise NotImplementedError(f"{engine=} not supported!")


novelty_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

novelty_prompt = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''


def check_idea_novelty(
        ideas,
        base_dir,
        platform,
        client,
        model_or_pipe,
        max_num_iterations=10,
        engine="openalex",
        show_thought=False,
):
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    

    for idx, idea in enumerate(ideas):
        try:
            if "novel" in idea:
                print(f"Skipping idea {idx}, already checked.")
                continue

            print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

            novel = False
            msg_history = []
            papers_str = ""

            for j in range(max_num_iterations):
                try:
                    text, msg_history = get_response_from_local_llm(
                        novelty_prompt.format(
                            current_round=j + 1,
                            num_rounds=max_num_iterations,
                            idea=idea,
                            last_query_results=papers_str,
                        ),
                        platform=platform,
                        client=client,
                        model_or_pipe=model_or_pipe,
                        system_message=novelty_system_msg.format(
                            num_rounds=max_num_iterations,
                            task_description=task_description,
                            code=code,
                        ),
                        msg_history=msg_history,
                        show_thought=show_thought,
                    )
                    if "decision made: novel" in text.lower():
                        print("Decision made: novel after round", j)
                        novel = True
                        break
                    if "decision made: not novel" in text.lower():
                        print("Decision made: not novel after round", j)
                        break

                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert json_output is not None, "Failed to extract JSON from LLM output"
                    ## SEARCH FOR PAPERS
                    query = json_output["Query"]
                    papers = search_for_papers(query, result_limit=10, engine=engine)
                    if papers is None:
                        papers_str = "No papers found."

                    paper_strings = []
                    for i, paper in enumerate(papers):
                        if engine == "core":
                            venue = "Unknown"
                            year = "Unknown"
                        else:
                            venue = paper["venue"]
                            year = paper["year"]

                        paper_strings.append(
                            """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                                i=i,
                                title=paper["title"],
                                authors=paper["authors"],
                                venue=venue,
                                year=year,
                                cites=paper["citationCount"],
                                abstract=paper["abstract"],
                            )
                        )
                    papers_str = "\n\n".join(paper_strings)

                except Exception as e:
                    #print(f"Error: {e}")
                    logging.exception("An unexpected error just happened.")
                    continue

            idea["novel"] = novel
        except Exception as e:
            logging.exception("An unexpected error just happened.")
            continue

    # Save results to JSON file
    results_file = osp.join(base_dir, "ideas.json")
    with open(results_file, "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


if __name__ == "__main__":
    import argparse
    import torch
    from transformers import pipeline
    from ai_scientist.llm import AVAILABLE_PLATFORMS, init_client_and_model_or_pipe
    
    MAX_NUM_GENERATIONS = 5
    NUM_REFLECTIONS = 3

    parser = argparse.ArgumentParser(description="Generate AI scientist ideas")
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="huggingface",
        choices=AVAILABLE_PLATFORMS,
        help="Model platform to use for AI Scientist.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Specify a name of your model to use from available platforms.",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="openalex",
        choices=["openalex", "core", "semanticscholar"],
        help="Scholar engine to use.",
    )
    args = parser.parse_args()

    client, model_or_pipe = init_client_and_model_or_pipe(args.platform, args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    ideas = generate_ideas(
        base_dir,
        args.platform,
        client,
        model_or_pipe,
        skip_generation=args.skip_idea_generation,
        max_num_generations=MAX_NUM_GENERATIONS,
        num_reflections=NUM_REFLECTIONS,
    )
    if not args.skip_novelty_check:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            platform=args.platform,
            client=client,
            model_or_pipe=model_or_pipe,
            engine=args.engine,
        )
