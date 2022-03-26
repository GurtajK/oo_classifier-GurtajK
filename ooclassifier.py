# ------------------------------
# Name(s): Gurtaj Khabra
# Cmput 274: Fall 2021
#
# Assignment 1: OO Classifier
# ------------------------------
"""
Object-Oriented Classifier (ooclassifier).  Base for Assignment #1, CMPUT 274.

Extended comments and functionality.

Copyright 2020-2021 Paul Lu
"""

import sys
import copy     # for deepcopy()

Debug = False   # Sometimes, print for debugging.  Overridable on command line.
InputFilename = "file.input.txt"
TargetWords = [
        'outside', 'today', 'weather', 'raining', 'nice', 'rain', 'snow',
        'day', 'winter', 'cold', 'warm', 'snowing', 'out', 'hope', 'boots',
        'sunny', 'windy', 'coming', 'perfect', 'need', 'sun', 'on', 'was',
        '-40', 'jackets', 'wish', 'fog', 'pretty', 'summer'
        ]


def open_file(filename=InputFilename):
    """
    Return an open file object or stdin for reading.
    Wrapper function for open() to handle common exceptions.
    Failed file open results in stdin used instead.

    Parameters
    ----------
    filename : string, default=InputFilename a global string literal
        Name/path to file.

    Returns
    -------
    file object
        Either a real file or stdin
    """
    try:
        f = open(filename, "r")
        return(f)
    except FileNotFoundError:
        # FileNotFoundError is subclass of OSError
        if Debug:
            print("File Not Found")
        return(sys.stdin)
    except OSError:
        if Debug:
            print("Other OS Error")
        return(sys.stdin)


def safe_input(f=None, prompt=""):
    """
    Return string with line of input, from file object or stdin, handling EOF

    Parameters
    ----------
    f : file object, default=None which causes stdin to be used
        File object or None for stdin
    prompt : string, default=""
        Optional prompt for input

    Returns
    -------
    tuple(string, boolean flag): string with line of input, flag=False means
        reached, otherwise True and string is line of input
    """
    try:
        # Case:  Stdin
        if f is sys.stdin or f is None:
            line = input(prompt)
        # Case:  From file
        else:
            assert not (f is None)
            assert (f is not None)
            line = f.readline()
            if Debug:
                print("readline: ", line, end='')
            if line == "":  # Check EOF before strip()
                if Debug:
                    print("EOF")
                return("", False)
        return(line.strip(), True)
    except EOFError:
        return("", False)


class C274:
    """
    Superclass for all classifier-related classes.

    Attributes
    ----------
    type : string
        Modifiable version of __class__

    Methods
    -------
    __init__
        Constructor, sets attribute "type"
    __str__
        Returns human-readable identification string
    __repr__
        Returns comparable identification string
    """
    def __init__(self):
        """
        Sets attribute "type"

        Returns
        -------
        None
        """
        self.type = str(self.__class__)
        return

    def __str__(self):
        """
        Returns human-readable identification string

        Returns
        -------
        string
            Returns human-readable identification string, currently "type"

        To do
        -----
        Better content than just attribute "type"
        """
        return(self.type)

    def __repr__(self):
        """
        Returns comparable identification string

        Returns
        -------
        string
            Returns comparable identification string, with "id" adn "type"
        """
        s = "<%d> %s" % (id(self), self.type)
        return(s)


class ClassifyByTarget(C274):
    def __init__(self, lw=[]):
        super().__init__()  # Call superclass
        # self.type = str(self.__class__)
        self.allWords = 0
        self.theCount = 0
        self.nonTarget = []
        self.set_target_words(lw)
        self.initTF()
        return

    def initTF(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        return

    # FIXME:  Incomplete.  Finish get_TF() and other getters/setters.
    def get_TF(self):
        return(self.TP, self.FP, self.TN, self.FN)

    # TODO: Could use Use Python properties
    #     https://www.python-course.eu/python3_properties.php
    def set_target_words(self, lw):
        # Could also do self.targetWords = lw.copy().  Thanks, TA Jason Cannon
        self.targetWords = copy.deepcopy(lw)
        return

    def get_target_words(self):
        return(self.targetWords)

    def get_allWords(self):
        return(self.allWords)

    def incr_allWords(self):
        self.allWords += 1
        return

    def get_theCount(self):
        return(self.theCount)

    def incr_theCount(self):
        self.theCount += 1
        return

    def get_nonTarget(self):
        return(self.nonTarget)

    def add_nonTarget(self, w):
        self.nonTarget.append(w)
        return

    def print_config(self, printSorted=True):
        print("-------- Print Config --------")
        ln = len(self.get_target_words())
        print("TargetWords (%d): " % ln, end='')
        if printSorted:
            print(sorted(self.get_target_words()))
        else:
            print(self.get_target_words())
        return

    def print_run_info(self, printSorted=True):
        print("-------- Print Run Info --------")
        print("All words:%3s. " % self.get_allWords(), end='')
        print(" Target words:%3s" % self.get_theCount())
        print("Non-Target words (%d): " % len(self.get_nonTarget()), end='')
        if printSorted:
            print(sorted(self.get_nonTarget()))
        else:
            print(self.get_nonTarget())
        return

    def print_confusion_matrix(self, targetLabel, doKey=False, tag=""):
        assert (self.TP + self.TP + self.FP + self.TN) > 0
        print(tag+"-------- Confusion Matrix --------")
        print(tag+"%10s | %13s" % ('Predict', 'Label'))
        print(tag+"-----------+----------------------")
        print(tag+"%10s | %10s %10s" % (' ', targetLabel, 'not'))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'TP   ', 'FP   '))
        print(tag+"%10s | %10d %10d" % (targetLabel, self.TP, self.FP))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'FN   ', 'TN   '))
        print(tag+"%10s | %10d %10d" % ('not', self.FN, self.TN))
        return

    def eval_training_set(self, tset, targetLabel, lines=True):
        print("-------- Evaluate Training Set --------")
        self.initTF()
        # zip is good for parallel arrays and iteration
        z = zip(tset.get_instances(), tset.get_lines())
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class()
            if lb == targetLabel:
                if cl:
                    self.TP += 1
                    outcome = "TP"
                else:
                    self.FN += 1
                    outcome = "FN"
            else:
                if cl:
                    self.FP += 1
                    outcome = "FP"
                else:
                    self.TN += 1
                    outcome = "TN"
            explain = ti.get_explain()
            # Format nice output
            if lines:
                w = ' '.join(w.split())
            else:
                w = ' '.join(ti.get_words())
                w = lb + " " + w

            # TW = testing bag of words words (kinda arbitrary)
            print("TW %s: ( %15s) %s" % (outcome, explain, w))
            if Debug:
                print("-->", ti.get_words())
        self.print_confusion_matrix(targetLabel)
        return

    def classify_by_words(self, ti, update=False, tlabel="last"):
        inClass = False
        evidence = ''
        lw = ti.get_words()
        for w in lw:
            if update:
                self.incr_allWords()
            if w in self.get_target_words():    # FIXME Write predicate
                inClass = True
                if update:
                    self.incr_theCount()
                if evidence == '':
                    evidence = w            # FIXME Use first word, but change
            elif w != '':
                if update and (w not in self.get_nonTarget()):
                    self.add_nonTarget(w)
        if evidence == '':
            evidence = '#negative'
        if update:
            ti.set_class(inClass, tlabel, evidence)
        return(inClass, evidence)

    # Could use a decorator, but not now
    def classify(self, ti, update=False, tlabel="last"):
        cl, e = self.classify_by_words(ti, update, tlabel)
        return(cl, e)

    def classify_all(self, ts, update=True, tlabel="classify_all"):
        for ti in ts.get_instances():
            cl, e = self.classify(ti, update=update, tlabel=tlabel)
        return


class ClassifyByTopN(ClassifyByTarget):
    """
    class summary

    Attributes
    ----------
    allWords : integer
    theCount : integer
    nonTarget : list of strings
        any word that does not appear in target words
    TP/FP/TN/FN : integer(s)
        the amount positive/negative classifications made
    targetWords : list of strings
        the target words being classified

    Methods
    -------
    __init__
        Constructor, sets attribute "type"
    target_top_n
        Sets target words to top N most frequent words
    """
    def __init__(self, lw=[]):
        """
        Sets same initilization as superclass

        Returns
        --------
        None
        """
        super().__init__()
        return

    def target_top_n(self, tset, num=5, label=''):
        """
        Iterates through all training instances in tset and counts
        the frequency of the words if label matches target label.
        Sets target words to the top N most frequent words found.

        Parameters
        -----------
        tset : trainingset object
            the training set object in which training
            instances are iterated through
        num : integer, default=5
            the number of words to target
        label : string
            the target label for which words are counted
        """
        wordsToCount = []  # list of words from instances that match label
        for ti in tset.get_instances():
            lb = ti.get_label()
            if lb == label:
                wordsToCount += ti.get_words()
        # dictionary with words as keys with counts as values
        wordFreq = dict()
        for w in wordsToCount:
            if w in wordFreq:
                # increase count by 1
                wordFreq[w] += 1
            else:
                # add word to wordFreq
                wordFreq[w] = 1
        counts = list(wordFreq.values())
        # a list of the top N words
        topN = []
        while len(topN) < num:
            if len(counts) == 0:
                break
            val = max(counts)
            for key, value in wordFreq.items():
                if value == val:
                    topN.append(key)
            while val in counts:
                counts.remove(val)
        # changes the target words attribute
        self.set_target_words(topN)
        return


class TrainingInstance(C274):
    def __init__(self):
        super().__init__()  # Call superclass
        # self.type = str(self.__class__)
        self.inst = dict()
        # FIXME:  Get rid of dict, and use attributes
        self.inst["label"] = "N/A"      # Class, given by oracle
        self.inst["words"] = []         # Bag of words
        self.inst["class"] = ""         # Class, by classifier
        self.inst["explain"] = ""       # Explanation for classification
        self.inst["experiments"] = dict()   # Previous classifier runs
        return

    def get_label(self):
        return(self.inst["label"])

    def get_words(self):
        return(self.inst["words"])

    def set_class(self, theClass, tlabel="last", explain=""):
        # tlabel = tag label
        self.inst["class"] = theClass
        self.inst["experiments"][tlabel] = theClass
        self.inst["explain"] = explain
        return

    def get_class_by_tag(self, tlabel):             # tlabel = tag label
        cl = self.inst["experiments"].get(tlabel)
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_explain(self):
        cl = self.inst.get("explain")
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_class(self):
        return self.inst["class"]

    def process_input_line(
                self, line, run=None,
                tlabel="read", inclLabel=False
            ):
        for w in line.split():
            if w[0] == "#":
                self.inst["label"] = w
                if inclLabel:
                    self.inst["words"].append(w)
            else:
                self.inst["words"].append(w)

        if not (run is None):
            cl, e = run.classify(self, update=True, tlabel=tlabel)
        return(self)

    def preprocess_words(self, mode=''):
        """
        Processeses all the words in this training instance.

        Parameters
        -----------
        mode : string, default=''
            Either 'keep-digits', 'keep-symbols', 'keep-stops', or none.

        Returns
        -----------
        none
        """
        # makes all words lower case
        for index in range(len(self.inst["words"])):
            self.inst["words"][index] = self.inst["words"][index].lower()
        if mode != 'keep-symbols':
            self.remove_symbols()
        if mode != 'keep-digits':
            self.remove_digits()
        if mode != 'keep-stops':
            self.remove_stops()
        return

    def remove_symbols(self):
        """
        Iterates through a string of all training instance words,
        removes symbols and reassigns as a list to training instance words.

        Returns
        ----------
        none
        """
        symbols = ''    # list of symbols that appear in given words
        # turns words into string
        originalString = ' '.join(self.inst["words"])
        for letter in originalString:
            if not letter.isalnum() and letter not in \
                    symbols and not letter == ' ':
                symbols += letter   # adds any non-alphanumeric char to symbols
        removeSymbols = originalString.maketrans('', '', symbols)
        newString = originalString.translate(removeSymbols)
        # previous 2 lines make new string which is original without symbols
        self.inst["words"] = newString.split()  # reclassify words
        return

    def remove_digits(self):
        """
        Iterates through all training instance words and removes digits.

        Returns
        ----------
        none
        """
        digits = ''
        for word in self.inst["words"]:
            for letter in word:
                if letter not in digits and letter.isnumeric():
                    digits += letter
        removeDigits = digits.maketrans('', '', digits)
        for index in range(len(self.inst["words"])):
            if not self.inst["words"][index].isnumeric():
                self.inst["words"][index] \
                    = self.inst["words"][index].translate(removeDigits)
        return

    def remove_stops(self):
        """
        Iterates through all training instance words
        and removes any stop words found.

        Returns
        ----------
        none
        """
        stopwords = ["i", "me", "my", "myself", "we", "our", "ours",
                     "ourselves", "you", "your", "yours", "yourself",
                     "yourselves", "he", "him", "his", "himself", "she",
                     "her", "hers", "herself", "it", "its", "itself", "they",
                     "them", "their", "theirs", "themselves", "what", "which",
                     "who", "whom", "this", "that", "these", "those", "am",
                     "is", "are", "was", "were", "be", "been", "being", "have",
                     "has", "had", "having", "do", "does", "did", "doing", "a",
                     "an", "the", "and", "but", "if", "or", "because", "as",
                     "until", "while", "of", "at", "by", "for", "with",
                     "about", "against", "between", "into", "through",
                     "during", "before", "after", "above", "below", "to",
                     "from", "up", "down", "in", "out", "on", "off", "over",
                     "under", "again", "further", "then", "once", "here",
                     "there", "when", "where", "why", "how", "all", "any",
                     "both", "each", "few", "more", "most", "other", "some",
                     "such", "no", "nor", "not", "only", "own", "same", "so",
                     "than", "too", "very", "s", "t", "can", "will", "just",
                     "don", "should", "now"]
        removeWords = []
        for word in self.inst["words"]:
            if word in stopwords:
                removeWords.append(word)
        for stop in removeWords:
            self.inst["words"].remove(stop)


class TrainingSet(C274):
    def __init__(self):
        super().__init__()  # Call superclass
        # self.type = str(self.__class__)
        self.inObjList = []     # Unparsed lines, from training set
        self.inObjHash = []     # Parsed lines, in dictionary/hash
        self.variable = dict()  # NEW: Configuration/environment variables
        return

    def set_env_variable(self, k, v):
        self.variable[k] = v
        return

    def get_env_variable(self, k):
        if k in self.variable:
            return(self.variable[k])
        else:
            return ""

    def inspect_comment(self, line):
        if len(line) > 1 and line[1] != ' ':      # Might be variable
            v = line.split(maxsplit=1)
            self.set_env_variable(v[0][1:], v[1])
        return

    def get_instances(self):
        return(self.inObjHash)      # FIXME Should protect this more

    def get_lines(self):
        return(self.inObjList)      # FIXME Should protect this more

    def print_training_set(self):
        print("-------- Print Training Set --------")
        z = zip(self.inObjHash, self.inObjList)
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class_by_tag("last")     # Not used
            explain = ti.get_explain()
            print("( %s) (%s) %s" % (lb, explain, w))
            if Debug:
                print("-->", ti.get_words())
        return

    def process_input_stream(self, inFile, run=None):
        assert not (inFile is None), "Assume valid file object"
        cFlag = True
        while cFlag:
            line, cFlag = safe_input(inFile)
            if not cFlag:
                break
            assert cFlag, "Assume valid input hereafter"

            if len(line) == 0:   # Blank line.  Skip it.
                continue

            # Check for comments *and* environment variables
            if line[0] == '%':  # Comments must start with % and variables
                self.inspect_comment(line)
                continue

            # Save the training data input, by line
            self.inObjList.append(line)
            # Save the training data input, after parsing
            ti = TrainingInstance()
            ti.process_input_line(line, run=run)
            self.inObjHash.append(ti)
        return

    def preprocess(self, mode=''):
        """
        Preprocess the words in each training instance.

        Parameters
        -----------
        mode : string
            Either 'keep-digits', 'keep-symbols', 'keep-stops', or none.
        """
        for ti in self.inObjHash:
            ti.preprocess_words(mode)
        return

    def return_nfolds(self, num=3):
        """
        Returns a list of num objects of class training set where
        each object is a portion of the original training set.

        Parameters
        -----------
        num : integers
            the number of partitions to make

        Retrns
        -----------
        nfolds : list of training set objects
            each object in the list contains a partition of the original set
        """
        nfolds = []
        for n in range(num):
            nfolds.append(TrainingSet())
            allInstances = self.get_instances()
            allLines = self.get_lines()
            index = n
            while index < len(allInstances):
                nfolds[n].inObjHash.append(copy.deepcopy(allInstances[index]))
                nfolds[n].inObjList.append(allLines[index])
                index += num
        return(nfolds)

    def copy(self):
        """
        Creates a deepcopy of this training set.

        Returns
        ---------
        copy.deepcopy(self)
        """
        return(copy.deepcopy(self))

    def add_training_set(self, tset):
        """
        Adds all training instances of tset to this training set.

        Parameters
        -----------
        tset : training set object
            The object which instances are copied from.
        """
        z = zip(tset.get_instances(), tset.get_lines())
        for ti, w in z:
            self.inObjHash.append(ti)
            self.inObjList.append(w)
        return


# Very basic test of functionality
def basemain():
    global Debug
    tset = TrainingSet()
    run1 = ClassifyByTarget(TargetWords)
    if Debug:
        print(run1)     # Just to show __str__
        lr = [run1]
        print(lr)       # Just to show __repr__

    argc = len(sys.argv)
    if argc == 1:   # Use stdin, or default filename
        inFile = open_file()
        assert not (inFile is None), "Assume valid file object"
        tset.process_input_stream(inFile, run1)
        inFile.close()
    else:
        for f in sys.argv[1:]:
            # Allow override of Debug from command line
            if f == "Debug":
                Debug = True
                continue
            if f == "NoDebug":
                Debug = False
                continue

            inFile = open_file(f)
            assert not (inFile is None), "Assume valid file object"
            tset.process_input_stream(inFile, run1)
            inFile.close()

    print("--------------------------------------------")
    plabel = tset.get_env_variable("pos-label")
    print("pos-label: ", plabel)
    print("NOTE: Not using any target words from the file itself")
    print("--------------------------------------------")

    if Debug:
        tset.print_training_set()
    run1.print_config()
    run1.print_run_info()
    run1.eval_training_set(tset, plabel)

    return


if __name__ == "__main__":
    basemain()
