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

import hashlib
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import regex as re


class LogCluster:
    """Represents a cluster of log messages that share a common template.

    Attributes:
        log_template: A list of tokens forming the template for this cluster,
            where variable parts are replaced by the wildcard <*>.
        log_ids: A list of line IDs (1-based) of all log messages assigned
            to this cluster.
    """

    def __init__(
        self,
        log_template: Optional[list[str]] = None,
        log_ids: Optional[list[int]] = None,
    ) -> None:
        self.log_template = log_template if log_template is not None else []
        self.log_ids = log_ids if log_ids is not None else []


class PrefixTreeNode:
    """A node in the prefix tree (trie) used by the Drain algorithm.

    The tree is structured as follows:
    - Root (depth 0): children are keyed by sequence length.
    - Length layer (depth 1): children are keyed by the first tokens of
      the log sequence up to LogParser.depth.
    - Token layers (depth 2 … depth): children are keyed by individual
      tokens or the wildcard <*>.
    - Leaf nodes: children holds a list of LogCluster
      objects instead of a dict.

    Attributes:
        children: Either a dict mapping a token / length to the next
            PrefixTreeNode, or a list of LogCluster objects at leaf 
            level.
        depth: Depth of this node in the tree (root = 0).
        token: The token or sequence-length value that labels the edge
            leading to this node (None for the root).
    """

    def __init__(
        self,
        children: Optional[dict] = None,
        depth: int = 0,
        token: Optional[str | int] = None,
    ) -> None:
        self.children = children if children is not None else {}
        self.depth = depth
        self.token = token


class LogParser:
    """Drain: an online log parsing algorithm based on a fixed-depth prefix tree.

    Reference:
        Pinjia He et al., "Drain: An Online Log Parsing Approach with
        Fixed Depth Tree", ICWS 2017.

    Parameters:
        log_format: A format string that describes the structure of each raw
            log line, e.g. "<Date> <Time> <Level> <Content>".  Field names 
            must be wrapped in <…>.
        input_dir: Directory that contains the raw log file(s).
        output_dir: Directory where structured CSV results will be written.
        tree_depth: Depth of the prefix tree (including the root and length
            layer).  The effective token-matching depth is tree_depth - 2.
        sim_threshold: Minimum cosine-like similarity score in [0, 1]
            required to merge a log message into an existing cluster.
        max_children: Maximum number of children per internal tree node.
            When the limit is reached, a wildcard <*> child is created
            instead.
        preprocessor_patterns: A list of regular-expression strings applied
            to each log message before parsing.  Every match is replaced
            with <*>.
        keep_parameters: If True, a ParameterList column containing the 
            extracted variable values is appended to the structured CSV.
    """

    def __init__(
        self,
        log_format: str,
        input_dir: str = "./",
        output_dir: str = "./result/",
        tree_depth: int = 4,
        sim_threshold: float = 0.4,
        max_children: int = 100,
        preprocessor_patterns: Optional[list[str]] = None,
        keep_parameters: bool = True,
    ) -> None:
        self.input_dir= input_dir
        self.tree_depth = tree_depth - 2
        self.sim_threshold = sim_threshold
        self.max_children = max_children
        self.log_filename = None
        self.output_dir = output_dir
        self.df_log = None
        self.log_format = log_format
        self.preprocessor_patterns = (
            preprocessor_patterns if preprocessor_patterns is not None else []
        )
        self.keep_parameters = keep_parameters

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_digits(self, token: str) -> bool:
        """Return True if token contains at least one digit character."""
        return any(char.isdigit() for char in token)

    # ------------------------------------------------------------------
    # Prefix-tree operations
    # ------------------------------------------------------------------

    def _tree_search(
        self, root: PrefixTreeNode, token_sequence: list[str]
    ) -> Optional[LogCluster]:
        """Search the prefix tree for the best-matching cluster.

        Traverses the tree guided by token_sequence and delegates to
        _fast_match at the leaf level.

        Parameters:
            root: Root node of the prefix tree.
            token_sequence: Tokenised, pre-processed log message.

        Returns:
            The best-matching LogCluster if similarity exceeds
            sim_threshold, otherwise None.
        """
        seq_len = len(token_sequence)
        if seq_len not in root.children:
            return None

        current_node = root.children[seq_len]

        depth = 1
        for token in token_sequence:
            if depth >= self.tree_depth or depth > seq_len:
                break

            if token in current_node.children:
                current_node = current_node.children[token]
            elif "<*>" in current_node.children:
                current_node = current_node.children["<*>"]
            else:
                return None

            depth += 1

        candidate_clusters = current_node.children
        return self._fast_match(candidate_clusters, token_sequence)

    def _add_cluster_to_tree(
        self, root: PrefixTreeNode, cluster: LogCluster
    ) -> None:
        """Insert cluster into the prefix tree rooted at root.

        Creates intermediate PrefixTreeNode objects as needed,
        respecting max_children and replacing numeric tokens with
        the <*> wildcard.

        Parameters:
            root: Root node of the prefix tree.
            cluster: The new cluster whose template drives the insertion path.
        """
        seq_len = len(cluster.log_template)
        if seq_len not in root.children:
            length_node = PrefixTreeNode(depth=1, token=seq_len)
            root.children[seq_len] = length_node
        else:
            length_node = root.children[seq_len]

        current_node = length_node
        depth = 1

        for token in cluster.log_template:
            # Reached the desired depth — attach the cluster to this leaf.
            if depth >= self.tree_depth or depth > seq_len:
                if len(current_node.children) == 0:
                    current_node.children = [cluster]
                else:
                    current_node.children.append(cluster)
                break

            if token not in current_node.children:
                if not self._has_digits(token):
                    if "<*>" in current_node.children:
                        if len(current_node.children) < self.max_children:
                            new_node = PrefixTreeNode(depth=depth + 1, token=token)
                            current_node.children[token] = new_node
                            current_node = new_node
                        else:
                            current_node = current_node.children["<*>"]
                    else:
                        if len(current_node.children) + 1 < self.max_children:
                            new_node = PrefixTreeNode(depth=depth + 1, token=token)
                            current_node.children[token] = new_node
                            current_node = new_node
                        elif len(current_node.children) + 1 == self.max_children:
                            wildcard_node = PrefixTreeNode(depth=depth + 1, token="<*>")
                            current_node.children["<*>"] = wildcard_node
                            current_node = wildcard_node
                        else:
                            current_node = current_node.children["<*>"]
                else:
                    if "<*>" not in current_node.children:
                        wildcard_node = PrefixTreeNode(depth=depth + 1, token="<*>")
                        current_node.children["<*>"] = wildcard_node
                        current_node = wildcard_node
                    else:
                        current_node = current_node.children["<*>"]
            else:
                current_node = current_node.children[token]

            depth += 1

    # ------------------------------------------------------------------
    # Similarity / matching
    # ------------------------------------------------------------------

    def _sequence_distance(
        self, template: list[str], sequence: list[str]
    ) -> tuple[float, int]:
        """Compute the similarity between a template and a log sequence.

        Counts non-wildcard positions where both sequences share the same
        token and divides by the total sequence length.

        Parameters:
            template: Token list for the existing cluster template.
                Wildcards are represented as <*>.
            sequence: Token list of the incoming log message.

        Returns:
            A tuple (similarity, wildcard_count) where similarity is in [0, 1] 
            and wildcard_count is the number of <*> placeholders in template.
        """
        assert len(template) == len(sequence)

        matching_tokens = 0
        wildcard_count = 0

        for t1, t2 in zip(template, sequence):
            if t1 == "<*>":
                wildcard_count += 1
                continue
            if t1 == t2:
                matching_tokens += 1

        similarity = float(matching_tokens) / len(template)
        return similarity, wildcard_count

    def _fast_match(
        self, clusters: list[LogCluster], sequence: list[str]
    ) -> Optional[LogCluster]:
        """Select the best-matching cluster from a leaf's cluster list.

        Among all clusters whose similarity to sequence meets sim_threshold,
        the one with the highest similarity is returned; ties are broken by
        preferring the cluster with more wildcard positions.

        Parameters:
            clusters: List of LogCluster objects stored at a prefix-tree leaf 
                node.
            sequence: Tokenised log message to match.

        Returns:
            The best-matching LogCluster, or None if no cluster exceeds the 
                threshold.
        """
        best_cluster = None
        best_similarity = -1.0
        best_wildcard_count = -1

        for cluster in clusters:
            similarity, wildcard_count = self._sequence_distance(
                cluster.log_template, sequence
            )
            if similarity > best_similarity or (
                similarity == best_similarity and wildcard_count > best_wildcard_count
            ):
                best_similarity = similarity
                best_wildcard_count = wildcard_count
                best_cluster = cluster

        if best_similarity >= self.sim_threshold:
            return best_cluster
        return None

    def _merge_templates(
        self, new_sequence: list[str], existing_template: list[str]
    ) -> list[str]:
        """Produce an updated template by merging new_sequence into existing_template.

        Token positions where the two sequences differ are replaced with
        the wildcard <*>.

        Parameters:
            new_sequence: Tokenised log message.
            existing_template: Current template of the matched cluster.

        Returns:
            A new token list representing the merged template.
        """
        assert len(new_sequence) == len(existing_template)

        merged = []
        for token_new, token_existing in zip(new_sequence, existing_template):
            if token_new == token_existing:
                merged.append(token_new)
            else:
                merged.append("<*>")

        return merged

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_results(self, clusters: list[LogCluster]) -> None:
        """Persist parsing results to disk as two CSV files.

        Writes:
        - <log_filename>_structured.csv — original log data enriched with
          EventId, EventTemplate, and optionally ParameterList columns.
        - <log_filename>_templates.csv — deduplicated event templates with
          their IDs and occurrence counts.

        Parameters:
            clusters: All LogCluster objects produced by the parse.
        """
        log_templates = [0] * self.df_log.shape[0]
        log_template_ids = [0] * self.df_log.shape[0]

        for cluster in clusters:
            template_str = " ".join(cluster.log_template)
            template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[:8]
            for log_id in cluster.log_ids:
                row_idx = log_id - 1
                log_templates[row_idx] = template_str
                log_template_ids[row_idx] = template_id

        self.df_log["EventId"] = log_template_ids
        self.df_log["EventTemplate"] = log_templates

        if self.keep_parameters:
            self.df_log["ParameterList"] = self.df_log.apply(
                self._extract_parameters, axis=1
            )

        structured_path = os.path.join(
            self.output_dir, self.log_filename + "_structured.csv"
        )
        self.df_log.to_csv(structured_path, index=False)

        occurrence_counts = dict(self.df_log["EventTemplate"].value_counts())
        df_events = pd.DataFrame()
        df_events["EventTemplate"] = self.df_log["EventTemplate"].unique()
        df_events["EventId"] = df_events["EventTemplate"].map(
            lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()[:8]
        )
        df_events["Occurrences"] = df_events["EventTemplate"].map(occurrence_counts)

        templates_path = os.path.join(
            self.output_dir, self.log_filename + "_templates.csv"
        )
        df_events.to_csv(
            templates_path,
            index=False,
            columns=["EventId", "EventTemplate", "Occurrences"],
        )

    def print_tree(self, node: PrefixTreeNode, indent: int = 0) -> None:
        """Recursively print the prefix tree for debugging purposes.

        Parameters:
            node: The node to print (start with the root).
            indent: Current indentation level (increases with depth).
        """
        prefix = "\t" * indent

        if node.depth == 0:
            label = "Root"
        elif node.depth == 1:
            label = f"<{node.token}>"
        else:
            label = str(node.token)

        print(prefix + label)

        if node.depth == self.tree_depth:
            return

        for child in node.children:
            self.print_tree(node.children[child], indent + 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, log_filename: str) -> None:
        """Parse a single log file and write structured output to output_dir.

        The method loads the file, iterates over every log line, searches
        the prefix tree for a matching cluster, and either creates a new
        cluster or updates the best-matching one.  Results are written via
        _write_results upon completion.

        Parameters:
            log_filename: Name of the log file located inside input_dir.
        """
        log_path = os.path.join(self.input_dir, log_filename)
        print(f"Parsing file: {log_path}")

        start_time = datetime.now()
        self.log_filename = log_filename

        root_node = PrefixTreeNode()
        all_clusters: list[LogCluster] = []

        self._load_data()

        processed = 0
        total = len(self.df_log)

        for _, row in self.df_log.iterrows():
            log_id: int = row["LineId"]
            tokens: list[str] = self._preprocess(row["Content"]).strip().split()

            matched_cluster = self._tree_search(root_node, tokens)

            if matched_cluster is None:
                new_cluster = LogCluster(log_template=tokens, log_ids=[log_id])
                all_clusters.append(new_cluster)
                self._add_cluster_to_tree(root_node, new_cluster)
            else:
                updated_template = self._merge_templates(tokens, matched_cluster.log_template)
                matched_cluster.log_ids.append(log_id)
                if " ".join(updated_template) != " ".join(matched_cluster.log_template):
                    matched_cluster.log_template = updated_template

            processed += 1
            if processed % 1000 == 0 or processed == total:
                print(f"Processed {processed * 100.0 / total:.1f}% of log lines.")

        os.makedirs(self.output_dir, exist_ok=True)
        self._write_results(all_clusters)

        print(f"Parsing done. [Time taken: {datetime.now() - start_time!s}]")

    # ------------------------------------------------------------------
    # Data loading & preprocessing
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load and parse the raw log file into df_log.

        Delegates format detection to _build_format_regex and line
        parsing to _log_to_dataframe.
        """
        headers, regex = self._build_format_regex(self.log_format)
        self.df_log = self._log_to_dataframe(
            os.path.join(self.input_dir, self.log_filename),
            regex,
            headers,
        )

    def _preprocess(self, line: str) -> str:
        """Apply all preprocessor_patterns to line.

        Each regex match is replaced with <*> so that variable parts
        (e.g. IP addresses, timestamps) do not fragment cluster templates.

        Parameters:
            line: A single raw log message (Content field).

        Returns:
            The pre-processed log message.
        """
        for pattern in self.preprocessor_patterns:
            line = re.sub(pattern, "<*>", line)
        return line

    def _log_to_dataframe(
        self,
        log_file: str,
        regex: re.Pattern,
        headers: list[str],
    ) -> pd.DataFrame:
        """Read log_file and return its contents as a pandas.DataFrame.

        Lines that do not match regex are skipped with a warning.

        Parameters:
            log_file: Absolute path to the raw log file.
            regex: Compiled regular expression with named groups corresponding
                to headers.
            headers: Ordered list of field names to extract from each line.

        Returns:
            A DataFrame with a LineId column (1-based) followed by one
            column per header.
        """
        log_messages: list[list[str]] = []
        line_count = 0

        with open(log_file, "r") as f:
            for line in f:
                try:
                    match = regex.search(line.strip())
                    message = [match.group(h) for h in headers]
                    log_messages.append(message)
                    line_count += 1
                except Exception:
                    print(f"[Warning] Skipping line: {line}")

        df = pd.DataFrame(log_messages, columns=headers)
        df.insert(0, "LineId", range(1, line_count + 1))
        print(f"Total lines: {len(df)}")
        return df

    def _build_format_regex(
        self, log_format: str
    ) -> tuple[list[str], re.Pattern]:
        """Convert a human-readable log_format string into a compiled regex.

        Format fields wrapped in <…> become named capture groups; literal
        text between fields is converted to \\s+ patterns.

        Parameters:
            log_format: Format descriptor such as
                "<Date> <Time> <Level> <Content>".

        Returns:
            A tuple (headers, pattern) where headers is the ordered
            list of field names and pattern is the compiled regex.
        """
        headers: list[str] = []
        splitters = re.split(r"(<[^<>]+>)", log_format)
        regex_str = ""

        for idx, part in enumerate(splitters):
            if idx % 2 == 0:
                # Literal text: normalise whitespace.
                regex_str += re.sub(r" +", r"\\s+", part)
            else:
                # Named field.
                field_name = part.strip("<>")
                regex_str += f"(?P<{field_name}>.*?)"
                headers.append(field_name)

        pattern = re.compile("^" + regex_str + "$")
        return headers, pattern

    def _extract_parameters(self, row: pd.Series) -> list[str]:
        """Extract the variable values captured by a log template.

        Builds a regex from the event template by replacing <*>
        wildcards with capture groups and matches it against the raw
        Content field.

        Parameters:
            row: A single row of df_log containing at least
                EventTemplate and Content columns.

        Returns:
            A list of extracted parameter strings (may be empty if the
            template contains no wildcards or no match is found).
        """
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []

        template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
        template_regex = re.sub(r"\\ +", r"\\s+", template_regex)
        template_regex = "^" + template_regex.replace(r"\<\*\>", "(.*?)") + "$"

        matches = re.findall(template_regex, row["Content"])
        if not matches:
            return []

        result = matches[0]
        return list(result) if isinstance(result, tuple) else [result]
