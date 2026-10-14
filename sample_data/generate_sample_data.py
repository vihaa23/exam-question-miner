"""
generate_sample_data.py — Create synthetic exam paper text files for demo purposes.

Run this script once to populate the sample_data/ directory with realistic
mock exam papers across 6 years (2019-2024) for a "Data Structures" course.
These are plain-text files with .txt extension — the pipeline treats them
exactly like PDF text (you can also supply actual PDFs).
"""

import os
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent
SAMPLE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Template exam papers (6 years, realistic question overlap)
# ─────────────────────────────────────────────────────────────────

PAPERS: dict[str, str] = {

"DS_2019.txt": """
University Examinations — Data Structures (CS201)
Time: 3 Hours     Maximum Marks: 100

Attempt any FIVE questions. All questions carry equal marks.

Q.1 Explain the concept of a stack and describe its operations (push, pop, peek). [10]
Q.2 What is a linked list? Differentiate between singly and doubly linked lists with suitable examples. [10]
Q.3 Write an algorithm for binary search. What is its time complexity? [10]
Q.4 Define a queue and explain its applications in operating systems. [10]
Q.5 Explain Bubble Sort with an example. Compare it with Insertion Sort. [10]
Q.6 What is a binary tree? Explain in-order, pre-order, and post-order traversal with diagrams. [10]
Q.7 Describe the concept of hashing. What are collision resolution techniques? [10]
""",

"DS_2020.txt": """
University Examinations — Data Structures (CS201)
Time: 3 Hours     Maximum Marks: 100

Attempt any FIVE questions.

Q.1 Describe the working of a stack using an array. Write a program to implement push and pop operations. [10]
Q.2 What is recursion? Explain with the help of a factorial program. [10]
Q.3 Explain binary search tree (BST). Write algorithms for insertion and deletion in a BST. [10]
Q.4 What is a queue? Differentiate between a simple queue and a circular queue. [10]
Q.5 Explain the merge sort algorithm with an example. What is its time complexity? [10]
Q.6 What are graphs? Explain BFS and DFS traversal algorithms. [10]
Q.7 Describe open addressing and chaining as collision resolution techniques in hashing. [10]
""",

"DS_2021.txt": """
University Examinations — Data Structures (CS201)
Time: 3 Hours     Maximum Marks: 100

Answer any FIVE of the following questions. Each question carries 10 marks.

Q.1 Explain the LIFO principle used in stacks. Implement a stack using a linked list. [10]
Q.2 What is a doubly linked list? Write algorithms to insert and delete nodes. [10]
Q.3 Describe quick sort. Analyse its best, worst, and average case time complexity. [10]
Q.4 Explain the concept of a priority queue and its use in scheduling algorithms. [10]
Q.5 What is a binary search? Write a recursive algorithm and explain with an example. [10]
Q.6 Explain AVL trees. How does AVL rotation maintain balance? [10]
Q.7 What is dynamic programming? Solve the 0/1 knapsack problem using dynamic programming. [10]
""",

"DS_2022.txt": """
University Examinations — Data Structures (CS201)
Time: 3 Hours     Maximum Marks: 100

All questions carry equal marks. Attempt any FIVE.

Q.1 What is a stack? Explain its applications in expression evaluation and backtracking. [10]
Q.2 Write and explain the algorithm for merge sort. Show its working on the array [5, 3, 8, 1, 9]. [10]
Q.3 Explain the structure of a binary search tree. Write the search algorithm. [10]
Q.4 What is hashing? Explain the hash function and collision resolution by separate chaining. [10]
Q.5 Differentiate between a singly linked list and a doubly linked list. [10]
Q.6 Explain DFS and BFS graph traversal with an example graph. [10]
Q.7 What is the concept of a heap? Explain max-heap and min-heap with examples. [10]
""",

"DS_2023.txt": """
University Examinations — Data Structures (CS201)
Time: 3 Hours     Maximum Marks: 100

Attempt any FIVE questions. All questions carry 10 marks each.

Q.1 Describe the stack data structure. Write a C program to evaluate a postfix expression using a stack. [10]
Q.2 What is a circular linked list? Write an algorithm to insert a node in a circular linked list. [10]
Q.3 Explain heap sort. Write the algorithm and analyse its time complexity. [10]
Q.4 Explain binary search with a suitable example. Compare binary search with linear search. [10]
Q.5 What is a graph? Define adjacency matrix and adjacency list representations. [10]
Q.6 Explain the AVL tree and the types of rotations used to rebalance it. [10]
Q.7 What is hashing? Describe open addressing (linear probing, quadratic probing). [10]
""",

"DS_2024.txt": """
University Examinations — Data Structures (CS201)
Time: 3 Hours     Maximum Marks: 100

Attempt any FIVE questions out of seven.

Q.1 Explain the concept of a stack and its operations. Implement a stack using arrays. [10]
Q.2 What is a binary search tree? Write algorithms for inserting, searching, and deleting a node. [10]
Q.3 Describe quick sort with an example. Discuss its advantages and disadvantages. [10]
Q.4 Explain the doubly linked list. Write an algorithm for deletion at a given position. [10]
Q.5 What is graph traversal? Explain BFS and DFS with examples. [10]
Q.6 Explain hashing and different collision handling strategies. [10]
Q.7 What is dynamic programming? Explain the longest common subsequence (LCS) problem. [10]
""",
}


def generate() -> None:
    """Write all sample papers to the sample_data/ directory."""
    for filename, content in PAPERS.items():
        path = SAMPLE_DIR / filename
        path.write_text(content.strip(), encoding="utf-8")
        print(f"  Created: {path}")
    print(f"\nDone! {len(PAPERS)} sample papers written to {SAMPLE_DIR}")


if __name__ == "__main__":
    generate()
