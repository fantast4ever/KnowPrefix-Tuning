import collections
import json
import os
from typing import List

cmu_dog_multi_msg_delimiter = " "
cmu_dog_only_with_knowledge = True
cmu_dog_fact_delimiter = ";"
cmu_dog_include_knowledge_keys = 'cast,critical_response,director,genre,introduction,movieName,rating,year'
cmu_dog_provide_movie_context = False

SILENCE = '__SILENCE__'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"]  


def _collapse_multi_msgs(history, multi_msg_delim):
    """
    This dataset allows for a single user to send multiple messages in a row.

    Here we use a delimiter to represent this, like: "Hey!|Nice to meet you."
    """
    collapsed = []
    last_msg = history[0]
    for msg in history[1:]:
        if last_msg["uid"] == msg["uid"]:
            last_msg["text"] = multi_msg_delim.join((last_msg["text"], msg["text"]))
        else:
            collapsed.append(last_msg)
            last_msg = msg
    
    collapsed.append(last_msg)
    return collapsed


def _article_section_to_text(
    section, fact_delimiter: str, knowledge_keys: List[str] = None
) -> str:
    """
    Example input:
    {
      "cast": [
        "Ben Affleck as Batman",
        "Henry Cavill as Superman",
      ],
      "director": "Zack Snyder"
    }
    Example output:
    "cast:Ben Affleck as Batman,Henry Cavill as Superman;director:Zack Snyder"
    """
    if not section:
        return section
    if isinstance(section, str):
        return section
    texts = []
    for k, v in section.items():
        if knowledge_keys and k not in knowledge_keys:
            continue
        fact = f"{k}:"
        if isinstance(v, str):
            fact += v
        else:
            fact += ",".join(v)
        texts.append(fact)
    return fact_delimiter.join(texts)


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."


def load_data(in_dir, in_file):
    data_file = os.path.join(in_dir, in_file)
    wiki_data_file = os.path.join(in_dir, "wiki_data.json")

    with open(data_file, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    with open(wiki_data_file, mode="r", encoding="utf-8") as f:
        wiki_data = json.load(f)

    
    data = {
        k: c for k, c in data.items() if c["rating"] in [1, 2, 3]
    }

    def _can_see_info(turn, convo):
        
        return turn["uid"] in convo["whoSawDoc"]

    num_eps = len(data)
    data = list(data.items())
    examples = []
    
    for i in range(len(data) * 2):
        conv_idx = i % num_eps
        start_idx = i // num_eps

        _conv_id, conv_data = data[conv_idx]

        dialog = _collapse_multi_msgs(
            conv_data["history"], cmu_dog_multi_msg_delimiter
        )
        movie_article = wiki_data[str(conv_data["wikiDocumentIdx"])]

        if cmu_dog_only_with_knowledge and not _can_see_info(
                dialog[start_idx], conv_data
        ):
            continue

        
        for idx in range(start_idx, len(dialog), 2):
            episode_done = idx == len(dialog) - 1
            label_turn = dialog[idx]
            label = label_turn["text"].strip()

            
            doc_idx = str(label_turn["docIdx"])
            gold_knowledge = _article_section_to_text(
                movie_article[doc_idx], cmu_dog_fact_delimiter
            )
            section = (
                movie_article[doc_idx]
                if _can_see_info(label_turn, conv_data)
                else None
            )

            section_text = _article_section_to_text(
                section,
                cmu_dog_fact_delimiter,
                cmu_dog_include_knowledge_keys.split(','),
            )

            
            if idx == start_idx:
                context = (
                    section_text
                    if cmu_dog_provide_movie_context
                    else SILENCE
                )
            else:
                context = dialog[idx - 1]["text"].strip()

            examples.append({
                'text': context,
                'labels': [label],
                'title': movie_article['0']['movieName'],
                'checked_sentence': [gold_knowledge],
                'episode_done': episode_done
            })
    print('loaded total of {} examples'.format(len(examples)))
    return examples


def data_generator(in_dir, in_file, keep_last_n=99999):

    examples = load_data(in_dir, in_file)
    observation = None
    history_strings = []
    users = []

    reset_on_next_update = False

    for i, ex in enumerate(examples):
        if i % 1000 == 0:
            print("Processing {} of {}; {:0.2f} percent done".format(
                i, len(examples), float(i) * 100.0 / float(len(examples))))
        if not observation or observation['episode_done']:
            last_reply = None
        else:
            last_reply = observation['labels'][0].lower()

            last_reply = fix_missing_period(last_reply)
        observation = ex.copy()
        if reset_on_next_update:
            history_strings = []
            users = []
            reset_on_next_update = False

        if last_reply is not None:
            last_reply = last_reply.lower()
            history_strings.append(last_reply)
            users.append(1)

        if 'text' in observation and observation['text'] is not None:

            next_text = " ".join(observation['text'].split()).lower()
            next_text = fix_missing_period(next_text)
            history_strings.append(next_text)
            users.append(0)

        if observation['episode_done']:
            reset_on_next_update = True

        label = " ".join(observation['labels'][0].split()).lower()

        knowledge = observation["checked_sentence"]

        yield (history_strings[-keep_last_n:], label, knowledge)

