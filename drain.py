# =========================================================================
# Copyright (C) 2016-2023 LOGPAI (https://github.com/logpai).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import regex as re
import os
import pandas as pd
import hashlib
from datetime import datetime
from typing import Optional, Union


class Logcluster:
    """Represents a cluster of log messages that share a common template.

    Each cluster holds the inferred log template (a list of tokens, where
    variable parts are replaced by ``<*>``) and the IDs of all log lines
    that have been matched to it.

    Attributes:
        logTemplate: Tokenized log template, e.g. ``["node", "<*>", "failed"]``.
        logIDL: List of 1-based line IDs of log messages belonging to this cluster.
    """

    def __init__(
        self,
        logTemplate: Optional[list[str]] = None,
        logIDL: Optional[list[int]] = None,
    ) -> None:
        self.logTemplate: list[str] = logTemplate if logTemplate is not None else []
        self.logIDL: list[int] = logIDL if logIDL is not None else []


class Node:
    """A node in the prefix tree (trie) used by the Drain algorithm.

    The tree is structured as follows:

    * **Root** (depth 0): its children are keyed by sequence length.
    * **Length layer** (depth 1): one child per distinct token-count seen so far.
    * **Token layers** (depth 2 … ``self.depth``): each non-leaf node stores a
      mapping from token string (or ``"<*>"`` as a wildcard) to the next node.
    * **Leaf nodes** (depth == ``parser.depth``): ``childD`` is a plain
      :class:`list` of :class:`Logcluster` objects rather than a ``dict``.

    Attributes:
        childD: Either a ``dict`` mapping tokens/lengths to child :class:`Node`
            objects, or a ``list`` of :class:`Logcluster` objects when this
            node is a leaf.
        depth: Depth of this node in the tree (root is 0).
        digitOrtoken: The value used to reach this node from its parent —
            either a sequence length (int, depth 1) or a token string
            (str, depth ≥ 2).
    """

    def __init__(
        self,
        childD: Optional[dict] = None,
        depth: int = 0,
        digitOrtoken: Optional[Union[str, int]] = None,
    ) -> None:
        self.childD: Union[dict, list] = childD if childD is not None else {}
        self.depth: int = depth
        self.digitOrtoken: Optional[Union[str, int]] = digitOrtoken


class LogParser:
    """Drain: an online log parsing algorithm based on a fixed-depth prefix tree.

    The parser processes raw log lines one by one, building a lightweight
    prefix tree that groups similar messages into clusters. Each cluster is
    summarised by a *log template* — the common token skeleton shared by all
    messages in the cluster, with variable tokens replaced by ``<*>``.

    References:
        He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). Drain: An online log
        parsing approach with fixed depth tree. *ICWS 2017*.

    Attributes:
        path: Directory containing the input log file.
        depth: Effective prefix-tree depth (``depth`` parameter minus 2).
        st: Similarity threshold in [0, 1]. A candidate cluster is only
            accepted when its token-similarity to the incoming message meets
            or exceeds this value.
        maxChild: Maximum number of children allowed on any internal tree node
            before a wildcard (``<*>``) bucket is used instead.
        logName: Filename of the log file currently being parsed.
        savePath: Directory where structured output files are written.
        df_log: :class:`pandas.DataFrame` built from the raw log file after
            :meth:`load_data` is called.
        log_format: Log format string using ``<FieldName>`` placeholders, e.g.
            ``"<Date> <Time> <Level> <Content>"``.
        rex: List of regular-expression strings applied to each log message
            during preprocessing; matches are replaced with ``<*>``.
        keep_para: When ``True``, a ``ParameterList`` column is added to the
            structured output containing the values extracted from each log
            line according to its template.
    """

    def __init__(
        self,
        log_format: str,
        indir: str = "./",
        outdir: str = "./result/",
        depth: int = 4,
        st: float = 0.4,
        maxChild: int = 100,
        rex: Optional[list[str]] = None,
        keep_para: bool = True,
    ) -> None:
        """Initialise the Drain log parser.

        Args:
            log_format: Format string describing each log line's structure.
                Named fields are enclosed in angle brackets, e.g.
                ``"<Date> <Time> <Pid> <Level> <Component>: <Content>"``.
            indir: Path to the directory that contains the input log file.
            outdir: Path to the directory where parsed results are saved.
                Created automatically if it does not exist.
            depth: Total depth of the prefix tree (including the root and the
                length layer). Minimum effective value is 2. Higher values
                produce finer-grained matching at the cost of more memory.
            st: Similarity threshold for merging a log message into an
                existing cluster. Must be in [0.0, 1.0]. Lower values
                produce fewer, broader clusters; higher values yield more
                precise but numerous templates.
            maxChild: Maximum number of direct children per internal tree
                node. When this limit is reached, further tokens are routed
                through a ``<*>`` wildcard child.
            rex: Optional list of regular-expression patterns to apply during
                preprocessing. Each match in a log message is replaced with
                ``<*>`` before the message is fed into the tree.
            keep_para: If ``True``, the output CSV will contain a
                ``ParameterList`` column holding the runtime values
                extracted from each line by matching against its template.
        """
        self.path: str = indir
        self.depth: int = depth - 2
        self.st: float = st
        self.maxChild: int = maxChild
        self.logName: Optional[str] = None
        self.savePath: str = outdir
        self.df_log: Optional[pd.DataFrame] = None
        self.log_format: str = log_format
        self.rex: list[str] = rex if rex is not None else []
        self.keep_para: bool = keep_para

    def hasNumbers(self, s: str) -> bool:
        """Return ``True`` if *s* contains at least one digit character.

        Args:
            s: The string to inspect.

        Returns:
            ``True`` when *s* contains a digit; ``False`` otherwise.
        """
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn: Node, seq: list[str]) -> Optional[Logcluster]:
        """Search the prefix tree for the best-matching log cluster.

        Traverses the tree guided by the length of *seq* and its leading
        tokens, then delegates final selection to :meth:`fastMatch`.

        Args:
            rn: Root node of the prefix tree.
            seq: Tokenised and preprocessed log message to look up.

        Returns:
            The best-matching :class:`Logcluster` if one meets the similarity
            threshold, or ``None`` if no suitable cluster exists.
        """
        retLogClust = None

        seqLen = len(seq)
        if seqLen not in rn.childD:
            return retLogClust

        parentn = rn.childD[seqLen]

        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break

            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif "<*>" in parentn.childD:
                parentn = parentn.childD["<*>"]
            else:
                return retLogClust
            currentDepth += 1

        logClustL = parentn.childD

        retLogClust = self.fastMatch(logClustL, seq)

        return retLogClust

    def addSeqToPrefixTree(self, rn: Node, logClust: Logcluster) -> None:
        """Insert a new log cluster into the prefix tree.

        Creates any missing intermediate nodes along the path determined by
        the cluster's template tokens, respecting the ``maxChild`` limit at
        every level by using a ``<*>`` wildcard bucket when necessary.

        Args:
            rn: Root node of the prefix tree.
            logClust: The newly created cluster whose template defines the
                insertion path.
        """
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.logTemplate:
            # Add current log cluster to the leaf node
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            # If token not matched in this layer of existing tree.
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if "<*>" in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD["<*>"]
                    else:
                        if len(parentn.childD) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken="<*>")
                            parentn.childD["<*>"] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD["<*>"]

                else:
                    if "<*>" not in parentn.childD:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken="<*>")
                        parentn.childD["<*>"] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD["<*>"]

            # If the token is matched
            else:
                parentn = parentn.childD[token]

            currentDepth += 1

    def seqDist(
        self, seq1: list[str], seq2: list[str]
    ) -> tuple[float, int]:
        """Compute the similarity between a log template and a log message.

        Counts the number of tokens that are identical in both sequences
        (wildcard positions in *seq1* are skipped) and divides by the total
        sequence length to produce a similarity score in [0, 1].

        Args:
            seq1: The log template token list. ``<*>`` entries are treated as
                wildcards and do not contribute to the similarity score but
                are counted separately.
            seq2: The incoming log message token list. Must have the same
                length as *seq1*.

        Returns:
            A 2-tuple ``(similarity, num_wildcards)`` where *similarity* is
            the fraction of matching non-wildcard tokens and *num_wildcards*
            is the number of ``<*>`` positions in *seq1*.

        Raises:
            AssertionError: If *seq1* and *seq2* differ in length.
        """
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == "<*>":
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar

    def fastMatch(
        self, logClustL: list[Logcluster], seq: list[str]
    ) -> Optional[Logcluster]:
        """Select the best-matching cluster from a candidate list.

        Iterates over all clusters at a leaf node and returns the one with
        the highest similarity score (ties broken by preferring the cluster
        with more wildcard positions, which indicates a more general template).

        Args:
            logClustL: List of :class:`Logcluster` candidates stored at the
                reached leaf node.
            seq: Tokenised log message to match against.

        Returns:
            The best :class:`Logcluster` if its similarity meets or exceeds
            ``self.st``, otherwise ``None``.
        """
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.st:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(
        self, seq1: list[str], seq2: list[str]
    ) -> list[str]:
        """Merge two token sequences into an updated log template.

        Compares the incoming log message (*seq1*) with the existing template
        (*seq2*) token by token. Positions where the tokens differ are
        replaced with the ``<*>`` wildcard in the returned template.

        Args:
            seq1: Tokenised log message (the newly observed line).
            seq2: Current log template token list.

        Returns:
            A new token list representing the merged template, with ``<*>``
            at every position where *seq1* and *seq2* disagree.

        Raises:
            AssertionError: If *seq1* and *seq2* differ in length.
        """
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append("<*>")

            i += 1

        return retVal

    def outputResult(self, logClustL: list[Logcluster]) -> None:
        """Write the parsing results to disk.

        Produces two CSV files inside ``self.savePath``:

        * ``<logName>_structured.csv`` — the original log data frame augmented
          with ``EventId``, ``EventTemplate``, and (optionally)
          ``ParameterList`` columns.
        * ``<logName>_templates.csv`` — one row per unique template with its
          MD5-derived ``EventId`` and occurrence count.

        Args:
            logClustL: List of all :class:`Logcluster` objects produced by
                the parsing run.
        """
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        for logClust in logClustL:
            template_str = " ".join(logClust.logTemplate)
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(
            df_events, columns=["EventId", "EventTemplate", "Occurrences"]
        )
        self.df_log["EventId"] = log_templateids
        self.df_log["EventTemplate"] = log_templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(
                self.get_parameter_list, axis=1
            )
        self.df_log.to_csv(
            os.path.join(self.savePath, self.logName + "_structured.csv"), index=False
        )

        occ_dict = dict(self.df_log["EventTemplate"].value_counts())
        df_event = pd.DataFrame()
        df_event["EventTemplate"] = self.df_log["EventTemplate"].unique()
        df_event["EventId"] = df_event["EventTemplate"].map(
            lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8]
        )
        df_event["Occurrences"] = df_event["EventTemplate"].map(occ_dict)
        df_event.to_csv(
            os.path.join(self.savePath, self.logName + "_templates.csv"),
            index=False,
            columns=["EventId", "EventTemplate", "Occurrences"],
        )

    def printTree(self, node: Node, dep: int) -> None:
        """Print a human-readable representation of the prefix tree.

        Recursively traverses the tree from *node* downward, indenting each
        level with a tab character. Useful for debugging the tree structure.

        Args:
            node: The node to start printing from (typically the root).
            dep: Current indentation depth (pass ``0`` for the root).
        """
        pStr = ""
        for i in range(dep):
            pStr += "\t"

        if node.depth == 0:
            pStr += "Root"
        elif node.depth == 1:
            pStr += "<" + str(node.digitOrtoken) + ">"
        else:
            pStr += node.digitOrtoken

        print(pStr)

        if node.depth == self.depth:
            return 1
        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def parse(self, logName: str) -> None:
        """Parse a log file and write structured output to disk.

        This is the main entry point of the Drain algorithm. It loads the log
        file, processes each line through the prefix tree, and saves the
        results via :meth:`outputResult`.

        For each log line:

        1. The content field is preprocessed (``rex`` substitutions applied).
        2. The prefix tree is searched for an existing matching cluster.
        3. If no match is found, a new cluster is created and inserted.
        4. If a match is found, the cluster's template is updated (additional
           positions that differ are widened to ``<*>``).

        Args:
            logName: Filename of the log file to parse (relative to
                ``self.path``).
        """
        print("Parsing file: " + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []

        self.load_data()

        count = 0
        for idx, line in self.df_log.iterrows():
            logID = line["LineId"]
            logmessageL = self.preprocess(line["Content"]).strip().split()
            matchCluster = self.treeSearch(rootNode, logmessageL)

            # Match no existing log cluster
            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)

            # Add the new log message to the existing cluster
            else:
                newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                matchCluster.logIDL.append(logID)
                if " ".join(newTemplate) != " ".join(matchCluster.logTemplate):
                    matchCluster.logTemplate = newTemplate

            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print(
                    "Processed {0:.1f}% of log lines.".format(
                        count * 100.0 / len(self.df_log)
                    )
                )

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.outputResult(logCluL)

        print("Parsing done. [Time taken: {!s}]".format(datetime.now() - start_time))

    def load_data(self) -> None:
        """Load and parse the raw log file into ``self.df_log``.

        Calls :meth:`generate_logformat_regex` to compile a regex from
        ``self.log_format``, then uses :meth:`log_to_dataframe` to read the
        file and populate ``self.df_log``.
        """
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(
            os.path.join(self.path, self.logName), regex, headers, self.log_format
        )

    def preprocess(self, line: str) -> str:
        """Apply user-defined regex substitutions to a raw log message.

        Each pattern in ``self.rex`` is applied in order; every match is
        replaced with ``<*>`` so that variable tokens are normalised before
        the message is tokenised and fed into the prefix tree.

        Args:
            line: The raw log message string (typically the ``Content`` field).

        Returns:
            The preprocessed log message with all matched patterns replaced
            by ``<*>``.
        """
        for currentRex in self.rex:
            line = re.sub(currentRex, "<*>", line)
        return line

    def log_to_dataframe(
        self,
        log_file: str,
        regex: re.Pattern,
        headers: list[str],
        logformat: str,
    ) -> pd.DataFrame:
        """Read a raw log file and return its contents as a :class:`pandas.DataFrame`.

        Each line is matched against *regex*; lines that do not match are
        skipped with a warning. A 1-based ``LineId`` column is prepended to
        the resulting data frame.

        Args:
            log_file: Absolute path to the raw log file.
            regex: Compiled regular expression with named groups corresponding
                to *headers*, generated by :meth:`generate_logformat_regex`.
            headers: Ordered list of field names extracted from the log format
                string.
            logformat: The original log format string (kept for reference but
                not used directly in this method).

        Returns:
            A :class:`pandas.DataFrame` with one row per successfully parsed
            log line and one column per header, plus a leading ``LineId``
            column.
        """
        log_messages = []
        linecount = 0
        with open(log_file, "r") as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    print("[Warning] Skip line: " + line)
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, "LineId", None)
        logdf["LineId"] = [i + 1 for i in range(linecount)]
        print("Total lines: ", len(logdf))
        return logdf

    def generate_logformat_regex(
        self, logformat: str
    ) -> tuple[list[str], re.Pattern]:
        """Compile a log format string into a regular expression and field list.

        Splits *logformat* on ``<FieldName>`` placeholders to extract field
        names and builds a regex with a named capture group for each field.
        Runs of spaces in literal segments are replaced with ``\s+`` to
        handle variable whitespace.

        Args:
            logformat: Format string using ``<FieldName>`` placeholders, e.g.
                ``"<Date> <Time> <Level> <Content>"``.

        Returns:
            A 2-tuple ``(headers, regex)`` where *headers* is an ordered list
            of field name strings and *regex* is the compiled
            :class:`re.Pattern` that can be used to parse individual log lines.
        """
        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(" +", "\\\s+", splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip("<").strip(">")
                regex += "(?P<%s>.*?)" % header
                headers.append(header)
        regex = re.compile("^" + regex + "$")
        return headers, regex

    def get_parameter_list(self, row: pd.Series) -> list[str]:
        """Extract parameter values from a log line using its event template.

        Converts the event template into a regex by replacing ``<*>``
        wildcards with capture groups, then matches it against the raw log
        content to extract the dynamic parts.

        Args:
            row: A single row from ``self.df_log``, expected to contain at
                least the ``"EventTemplate"`` and ``"Content"`` fields.

        Returns:
            A list of strings containing the captured parameter values. Returns
            an empty list if the template has no wildcards or if no match is
            found.
        """
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
        template_regex = re.sub(r"\\ +", r"\\s+", template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = (
            list(parameter_list)
            if isinstance(parameter_list, tuple)
            else [parameter_list]
        )
        return parameter_list
