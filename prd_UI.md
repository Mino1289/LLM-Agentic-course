Here is the Product Requirements Document (PRD) translated into English, structured formally, and stripped of all emojis, specifically designed to be given to an AI development agent (like Cursor, Cline, or Devin).

---

# Product Requirements Document (PRD)

**Project:** RAG Financial Assistant Interface Migration (Streamlit to Next.js)
**Target:** AI Development Agent

## 1. Context and Objectives

The objective of this project is to migrate an existing application built with Streamlit (`app_rag.py`) to a modern web application based on Next.js. The underlying system is a financial assistant leveraging a multi-agent architecture (Hub-and-Spoke) and RAG.

The new interface must transition away from the "analytical dashboard" aesthetic of Streamlit and adopt a fluid, clean, and professional conversational interface, in line with current LLM market standards.

## 2. Required Technical Stack

* **Framework:** Next.js (App Router recommended)
* **UI Library:** React
* **Styling:** Tailwind CSS
* **UI Components:** shadcn/ui (or Tailwind-native equivalent)
* **Iconography:** lucide-react
* **Theme:** Native Dark Mode support using neutral shades (anthracite gray, black) with subtle color accents for primary actions.

## 3. Interface Architecture Specifications

### 3.1. General Layout

The application must be divided into two main sections:

* **Sidebar (Left):** A collapsible side panel containing navigation and meta-information.
* **Main Area (Right):** The chat window occupying the remainder of the screen, with a fixed input area at the bottom.

### 3.2. Sidebar

* **Primary Action:** A large "New conversation" button located at the top.
* **Navigation:** A scrollable list of recent conversations.
* **Sidebar Footer:**
* Subtle display of technical information (e.g., Chat: OpenAI/GPT-4, Embeddings: Cohere).
* A "Settings" button (gear icon) that triggers the opening of the configuration panel.



### 3.3. Configuration Panel (Settings)

This panel can be implemented as a central Modal or a right-side "Slide-over". It must replace the configuration currently found in the Streamlit sidebar and integrate Slider components for the following variables:

* Max chunks (4 to 12, default: 8)
* Sub-queries count (1 to 8, default: 2)
* Max price days (30 to 365, default: 180)
* Max price points (10 to 120, default: 40)
* Max price tickers (1 to 5, default: 3)
* Default price window (15 to 180, default: 90)
* Max agent/tool iterations (2 to 10, default: 6)
* **Available tools:** An Accordion component listing the agent's tools alongside their names and brief descriptions.

## 4. Chat Area Specifications

### 4.1. Standard Message Flow

* **User Messages:** Right-aligned, with a sleek style that contrasts slightly with the background.
* **Assistant Messages:** Left-aligned. Assistant messages must support Markdown rendering.
* **Input Box:** Fixed at the bottom, full-width, utilizing the placeholder text: "Finance question (e.g.: Compare MSFT and NVDA — 2024 SEC risks, 6-month perf...)". The send button must be integrated inside the text input field.

### 4.2. Rich UI Components (Artifacts)

The assistant returns structured data that must not be displayed as raw text. The agent must generate specific React components for these elements, integrated seamlessly below the textual response:

* **Agent's Thoughts:** An Accordion component displaying a numbered list of tools called by the system and a summary of their execution.
* **Consulted Sources:** An Accordion component displaying cards containing text excerpts. This component must include dropdown menus (Select) in the header to allow visual filtering by "Ticker" and "Section".
* **Generated Reports:** Compact Card components presenting a file icon, the report name, and a direct download button.
* **Stats and Debug:** A component displaying technical metrics (Used tokens, Chunks, LLM Iterations) formatted as a grid of badges or a minimalist table.

## 5. Critical Component: Trade Approval (Human-in-the-loop)

The application requires human validation for trading actions. This component must be meticulously designed to draw the user's attention within the conversation flow.

* **Visual Header:** Title "Human Approval Required" styled as an alert or highlight (using a specific border or nuanced background color).
* **Data Grid:** Clear display of the following fields: Ticker, Side, Quantity, Order type, Limit price, and Risk Level.
* **Details:** A collapsible block (Accordion) allowing the user to read the full justification for the trade.
* **Actions:** Two prominent action buttons:
* "Approve Trade" button (Primary/Success style).
* "Cancel" button (Destructive/Warning style).



## 6. Execution Directives for the AI Agent

1. **Initialization:** Set up the Next.js project with Tailwind CSS and configure the foundational components via shadcn/ui.
2. **Modularity:** Strictly separate the code for the main layout, the sidebar, and the chat logic into distinct components.
3. **Rich Components:** Create a dedicated directory for components related to assistant messages (e.g., `ThoughtsRenderer`, `SourcesRenderer`, `TradeApprovalCard`).
4. **Mocking:** Generate a mock initial state simulating a complete conversation—including tool calls, RAG context, and a trade approval request—to visually validate the entire UI flow before hooking it up to the Python backend.