import os
import random

from datasets import load_from_disk

from language_modeling_via_stochastic_processes.src import constants
from language_modeling_via_stochastic_processes.src.datasets import encoder


class CodeParrotTriplet(encoder.BaseDataset):

    def __init__(
            self,
            train,
            seed,
            config,
            all_dataset=None,
            tokenizer_name="GPT2",
    ):

        dir_path = constants.PATH2CODEPARROT
        if train:
            self.filepath = os.path.join(dir_path, "train")
        else:
            self.filepath = os.path.join(dir_path, "test")

        super().__init__(
            train=train,
            tokenizer_name=tokenizer_name,
            all_dataset=all_dataset,
            seed=seed,
            config=config
        )

    def _load_data(self):
        self.data = load_from_disk(self.filepath)

    def _set_section_names(self):
        self.section_names = ['question', 'solution', 'break_statement', 'class_statement', 'continue_statement',
                              'def_statement',
                              'elif_statement', 'else_statement', 'expression_statement', 'for_statement',
                              'if_statement', 'import_statement', 'return_statement', 'while_statement']
        self.map = {"if": "if_statement", 'break': 'break_statement',
                    'class': 'class_statement', "continue": 'continue_statement',
                    'def': "def_statement", 'elif': 'elif_statement', 'else': 'else_statement',
                    'expression': 'expression_statement', 'for': 'for_statement',
                    'import': 'import_statement', 'return': 'return_statement',
                    'while': 'while_statement'}
        # duplicates --> if, if_statement
        self.section_ids = ['[ {} ]'.format(name.upper()) for name in self.section_names]

    def _process_data(self):
        self.processed_data = []
        for doc_id in range(len(self.data)):
            doc_info = []
            sentence_counter = 0

            question = self.data[doc_id]['question'].replace(".\n", ". ").split(self.split_pattern)[:-1]
            if len(question) == 0:
                question = [self.data[doc_id]['question']]
            question[0] = self.section_ids[0] + " " + question[0]
            question = [_ + ' . ' for _ in question]

            solutions = self.data[doc_id]['labelled_solutions']

            all_sentences = []
            for solution_id, solution in enumerate(solutions):
                text = question
                text += [f"{self.section_ids[1]} {solution_id}" + " . "]
                for line in solution:
                    section_map = line[0]
                    if line[0] in self.map:
                        section_map = self.map[line[0]]
                    if section_map not in self.section_names:
                        break
                    section_id = self.section_names.index(section_map)
                    text += [self.section_ids[section_id] + " " + line[1] + " . "]
                all_sentences += text

                for sentence in all_sentences:
                    if not sentence:
                        continue
                    if sentence == ' . ':
                        continue
                    sentence_info = {
                        "sentence": sentence,
                        "sentence_id": sentence_counter,
                        "doc_id": doc_id
                    }
                    doc_info.append(sentence_info)
                    sentence_counter += 1

                # Track total number of sentences in a document
                for info in doc_info:
                    info['total_doc_sentences'] = sentence_counter

                self.processed_data += doc_info

        print("Example: ", self.processed_data[0])
        print("Example: ", self.processed_data[10])

    def __getitem__(self, index):
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # Check if index is start of a seq. If so -> +2
        if sentence_num == 0:
            index += 2
        if sentence_num == 1:
            index += 1

        # Update
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # TRIAL 2: Sample all random points, t, t', t''
        T = sentence_num
        # t is a random point in between
        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]['sentence']
        y_t = self.processed_data[index - T + t2]['sentence']
        y_T = self.processed_data[index]['sentence']

        t_ = t1
        t = t2

        total_doc = utterance['total_doc_sentences']
        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': t_,
            't': t,
            'T': T,
            'total_t': total_doc,
        }
        return result

    def __len__(self):
        return len(self.processed_data) - 1
