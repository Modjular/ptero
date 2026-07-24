# V1 Architecture Plan: Single-Agent Bio-Image Harness

## Overview
A lightweight, browser-based agent harness designed to help non-technical scientists perform simple bio-image analysis tasks (e.g., cell segmentation, colocalization). Execution happens entirely client-side via Pyodide. To minimize latency and engineering complexity, V1 utilizes a single, highly capable agent rather than a multi-agent swarm.

## Core Design Philosophy (Adapted from Agentic-J)
*   **Simplicity Over Orchestration:** A single agent manages both the conversational planning and the coding execution. This eliminates inter-agent latency and context-loss bottlenecks.
*   **Write ≠ Run (The Scratchpad Protocol):** Code is never pushed to the user's UI without being verified first. The agent must test its assumptions.
*   **Domain Knowledge Over General Coding:** The agent relies on a strict set of bio-image analysis rules (e.g., preference for `skimage`, handling empty segmentations gracefully) injected via system prompts.

## Environment Architecture
The harness relies on a **Dual-Pyodide** setup running in the browser:
1.  **Scratch Instance (Hidden):** Used exclusively by the agent for testing code, inspecting array shapes, and catching runtime errors invisibly.
2.  **User-Facing Instance (Visible):** The actual notebook/UI where the final, validated code is deposited for the user to see and run.

## Agent Structure: The Solo "Co-Scientist"
*   **Role:** End-to-end responsibility. Handles user interaction, biological parameter gathering, code generation, error handling, and Pyodide execution.
*   **Persona:** Helpful, inquisitive biologist who happens to write great Python. Assumes nothing about the biology without asking.
*   **Constraints:** 
    *   **Strict Recursion Cap:** Maximum of 3 attempts to fix errors in the scratchpad.
    *   **Context Protection:** If it fails 3 times, it must return to the user with a plain-language summary of the roadblock. It must *never* pollute the main chat history with intermediate stack traces or failed scratchpad attempts. 

## Required Tool Set
1.  `test_in_scratch(code: str, mock_data_shape: tuple)`
    *   *Function:* Executes code in the hidden Pyodide instance.
    *   *Returns:* stdout, stderr, and output variable shapes/types.
2.  `push_to_ui(code: str, cell_index: int)`
    *   *Function:* Inserts the finalized, verified code into the user's visible interface.
3.  `ask_user(question: str)`
    *   *Function:* Halts execution and explicitly bubbles a question up to the user (e.g., clarifying channel indices or filter radii).

## Execution Workflow Example
1.  **User Request:** "Count all nuclei in the DAPI channel."
2.  **Planning & Clarification:** The agent proposes a standard watershed pipeline and uses the `ask_user` tool (or a standard conversational turn) to ask the user for the rolling ball background subtraction radius.
3.  **User Response:** User provides the radius.
4.  **Drafting & Testing (Hidden Loop):** 
    *   The agent drafts the `skimage` code.
    *   Calls `test_in_scratch`.
    *   Iterates internally if it encounters a shape mismatch or import error (capped at 3 tries).
5.  **Delivery Phase:** Once the code runs cleanly in the scratchpad, the agent calls `push_to_ui` to deliver the working code block to the user's view, accompanied by a brief success message.