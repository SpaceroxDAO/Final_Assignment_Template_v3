# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Hugging Face Space application for evaluating and submitting agent-based solutions to a scoring API. The project allows users to:
1. Define custom agents that answer questions
2. Fetch questions from an evaluation API
3. Submit answers and receive scoring feedback

## Key Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Install system dependencies for OCR
# macOS: brew install tesseract
# Ubuntu: apt-get install tesseract-ocr
# Windows: Download from GitHub (UB-Mannheim/tesseract)

# Run the Gradio app locally
python app.py
```

### Linting and Testing
No specific linting or testing commands are defined in this project.

## Architecture Overview

### Main Components
- **app.py**: Main Gradio application that handles UI, agent execution, and API submission
- **BasicAgent class**: Template agent implementation (lines 13-20) that should be customized
- **run_and_submit_all function**: Core workflow that fetches questions, runs agent, and submits answers

### Key Integration Points
- Default API endpoint: `https://agents-course-unit4-scoring.hf.space`
- Questions fetched from: `/questions`
- Answers submitted to: `/submit`
- HF OAuth authentication required for submission

### Environment Variables
- `SPACE_ID`: Hugging Face Space ID (used to generate code repository link)
- `SPACE_HOST`: Hugging Face Space host URL

### Submission Flow
1. User authenticates with Hugging Face OAuth
2. Questions are fetched from the API
3. Agent processes each question
4. Answers are collected and submitted with username and code repository link
5. Score and results are displayed in the UI

## Critical Development Guidelines

### Core Principles
* Always read entire files. Otherwise, you don't know what you don't know, and will end up making mistakes, duplicating code that already exists, or misunderstanding the architecture.
* Commit early and often. When working on large tasks, your task could be broken down into multiple logical milestones. After a certain milestone is completed and confirmed to be ok by the user, you should commit it.
* Your internal knowledgebase of libraries might not be up to date. When working with any external library, you will look up the latest syntax and usage.
* Do not say things like: "x library isn't working so I will skip it". Generally, it isn't working because you are using the incorrect syntax or patterns.
* Always run linting after making major changes to identify syntax errors or incorrect method usage.
* Please organise code into separate files wherever appropriate, and follow general coding best practices about variable naming, modularity, function complexity, file sizes.
* Code is read more often than it is written, make sure your code is always optimised for readability
* Unless explicitly asked otherwise, the user never wants you to do a "dummy" implementation. Never do an implementation where you tell the user: "This is how it *would* look like". Just implement the thing.
* Whenever you are starting a new task, you should ask the user follow up questions if you do not have clarity, rather than making incorrect assumptions.
* Do not carry out large refactors unless explicitly instructed to do so.

### Task Planning Requirements
* When starting on a new task, you should first understand the current architecture, identify the files you will need to modify, and come up with a Plan.
* In the Plan, you will think through architectural aspects related to the changes you will be making, consider edge cases, and identify the best approach.
* Get your Plan approved by the user before writing a single line of code.
* If you are running into repeated issues with a given task, figure out the root cause instead of throwing random things at the wall.

### Quality Standards
* You are an incredibly talented and experienced polyglot with decades of experience in diverse areas such as software architecture, system design, development, UI & UX, copywriting, and more.
* When doing UI & UX work, make sure your designs are both aesthetically pleasing, easy to use, and follow UI / UX best practices.
* When you receive a task that is very large in scope or too vague, you will first try to break it down into smaller subtasks.

## GAIA Benchmark Level 1: Complete Developer Guide

### Overview
GAIA (General AI Assistants benchmark) Level 1 contains 146 questions out of 466 total. These are conceptually simple for humans yet challenging for AI. Key characteristics:
- Require minimal effort: no external tools, or at most one tool
- No more than ~5 steps of reasoning or action
- Real-world scenarios spanning personal tasks, science, and general knowledge
- Answers not found verbatim in training data or internet
- Expected solution: single lookup or simple reasoning

### Question Types and Solution Approaches

#### 1. Direct Lookup Questions (Web or Knowledge Base)
- **Pattern**: Single fact retrieval from a reliable source
- **Solution**: Navigate to specific website/database, extract one piece of information
- **Example**: "What was the enrollment count of the H. pylori acne trial on NIH site?"
- **Tool required**: Web browsing or knowledge base access

#### 2. File-Based Data Questions
- **Pattern**: Extract/compute information from attached file (Excel, CSV, PDF, etc.)
- **Solution**: Parse file, perform simple operations (sum, filter, search)
- **Example**: "What were total food sales from the attached Excel?"
- **Tool required**: File parsing capability

#### 3. Multimodal/Visual Questions
- **Pattern**: Interpret image or audio to extract detail
- **Solution**: OCR for text in images, transcription for audio
- **Example**: "What license plate number is shown in the image?"
- **Tool required**: Image OCR or speech-to-text

#### 4. Simple Reasoning and Calculation
- **Pattern**: Light analytical reasoning or math on given information
- **Solution**: One or two steps of logic/arithmetic
- **Example**: Recipe conversion calculations, unit transformations
- **Tool required**: Internal reasoning or basic math

#### 5. Classification/Identification Tasks
- **Pattern**: Categorize item or decide true/false condition
- **Solution**: Apply classification logic to input
- **Example**: "Is this product a medical device under EU regulations?"
- **Tool required**: Internal reasoning

### Critical Answer Format Requirements

#### General Format Rules
- Answers must be factoid, concise, and unambiguous
- ONLY provide the requested information with NO additional commentary
- Follow exact formatting specified in the question
- Output should match expected pattern EXACTLY for automatic evaluation

#### Specific Format Guidelines

**Numbers:**
- No thousands separators unless specified
- Include units ONLY if explicitly requested
- Use digits, not words (90, not "ninety")
- Follow decimal precision if specified (e.g., "two decimal places")
- Example: "$89706.00" not "$89,706" or "89706 dollars"

**Strings (words/names):**
- Short phrase or name without additional words
- No articles ("the", "a") unless part of proper name
- Proper capitalization for names
- No abbreviations unless specifically requested
- Example: "San Francisco" not "The city of San Francisco"

**Lists:**
- Comma-separated in single line (unless otherwise specified)
- Each item follows above rules for numbers/strings
- Exact number of items as requested
- Example: "Alice, Bob" not "Alice and Bob" or "Alice, Bob, and Carol"

**Dates:**
- MUST follow specified format exactly
- Example: If asked for MM/DD/YY, answer "08/11/16" not "August 11, 2016"

### Common Patterns to Recognize

1. **Source Specification**: Questions often specify exact source ("according to NIH website", "as listed on Wikipedia")
2. **Time Context**: Look for "as of [date]" or specific year references
3. **Format Instructions**: Pay attention to phrases like "Express in USD with two decimal places"
4. **Stable Answers**: Questions designed so answers don't change over time
5. **Unambiguous Wording**: "What/Which/When" rather than "Why/How"

### Scoring Criteria (CRITICAL)

1. **Exact Match Required**: Answer must match ground truth exactly after minor normalization
2. **No Partial Credit**: Either fully correct or wrong (0 or 1 scoring)
3. **Case/Whitespace Normalized**: Minor differences ignored, but content must match
4. **Format Compliance**: Wrong format = wrong answer, even if factually correct
5. **Single Answer Only**: Don't hedge or provide alternatives

### Implementation Strategy for BasicAgent

1. **Parse Question Type**: Identify which category the question falls into
2. **Extract Requirements**: Find format specifications and source requirements
3. **Execute Tool Use**: Use appropriate tool (web search, file read, etc.)
4. **Process Information**: Apply any calculations or transformations
5. **Format Answer**: Ensure exact compliance with format requirements
6. **Verify Format**: Double-check formatting before returning
7. **Return Clean Answer**: Provide ONLY the requested information

### Common Failure Points to Avoid

- Adding extra context or explanation to answers
- Using wrong date/number formats
- Including units when not requested (or omitting when required)
- Providing multiple answers when one is expected
- Misreading format specifications in questions
- Adding articles or extra words to simple answers
- Using synonyms instead of exact terms requested

### Testing Strategy

- Use validation set to test format compliance
- Implement format validation before answer submission
- Test each question type category separately
- Verify tool usage patterns work correctly
- Ensure answer extraction is precise

## CURRENT PRIORITIES (Post 1/20 Analysis)

### 🚨 Benchmark Results: 1/20 Correct
Based on analysis of the run with 1 correct answer out of 20, these are the critical priorities:

#### 1. CRITICAL: Implement File Attachment Handling
**Status**: ✅ COMPLETED
- Fixed file passing in `run_and_submit_all` - now passes `attached_files_metadata`
- Updated `_handle_file_based_data` and `_handle_multimodal_visual` to handle various structures
- Added comprehensive logging throughout
- Handles both dict and list format from API

#### 2. Implement YouTube Data API v3
**Status**: ✅ COMPLETED
- Implemented `_fetch_youtube_data` method
- Integrated into `_handle_direct_lookup` for YouTube URLs
- Fetches title, description, channel, tags, view count
- Falls back to HTML scraping if API unavailable

#### 3. Implement Audio Transcription (Google Cloud Speech-to-Text)
**Status**: ✅ COMPLETED
- Implemented `_transcribe_audio` method 
- Handles MP3 → WAV conversion with pydub
- Integrated into `_handle_multimodal_visual`
- Note: Requires proper Google Cloud credentials setup

#### 4. Refine LLM Prompts for Content Accuracy
**Status**: ✅ COMPLETED
- Updated `_reason_with_llm` prompt to explicitly handle semantic opposites
- Added rules: "opposite of left is right", not reversed spelling
- Added instruction to decode reversed text first, then answer
- This should fix the "thgir" vs "right" issue

#### 5. Refine String Formatting Logic
**Status**: ✅ COMPLETED
- Fixed case preservation for acronyms/codes in `_format_string`
- Added IOC code detection and preservation in `_apply_string_subformat`
- Updated surname extraction to handle multiple items (Tanaka, Yamaguchi)
- Added case pattern detection for codes (e.g., LIE stays as LIE)
- Improved proper name handling while preserving existing case for codes

#### 6. Libraries to Add
Add to `requirements.txt`:
- `google-api-python-client`
- `google-auth-httplib2`
- `google-auth-oauthlib`
- `google-cloud-speech`
- `pydub` (if needed for audio conversion)

### Target File: `app.py`
All modifications should be made to the `BasicAgent` class in `app.py`.

---

## Complex Reasoning Enhancements ✅

### LLM Reasoning Improvements
- **Complex Rule Detection**: Added detection for complex rule-following tasks
- **Specialized Prompt**: Created `_complex_reasoning_with_llm` for step-by-step rule application
- **Better Indicators**: Detects keywords like "botanical", "rule", "exclude", "follow"
- **Structured Approach**: Prompt guides LLM through rule extraction and systematic application

### Fallback Logic Enhancements
- **List Processing**: Enhanced `_extract_and_apply_logic` to handle lists with rules
- **Rule Patterns**: Added patterns for "remove", "exclude", "from list" operations
- **Domain Knowledge**: Basic botanical fruit recognition for grocery list questions
- **Pattern Matching**: Improved extraction of lists and rules from natural language

### Key Changes
1. **_reason_with_llm**: Routes complex tasks to specialized handler
2. **_complex_reasoning_with_llm**: New method with tailored prompts
3. **_extract_and_apply_logic**: Enhanced with list filtering capabilities
4. **Better Fallbacks**: More capable rule-based logic for when LLM fails

---

## Phase 2 Improvements Completed ✅

### Answer Content Improvements
- **LLM Reasoning**: Enhanced prompt to correctly handle semantic opposites
- **String Formatting**: Improved case preservation and multiple item handling
- **List Processing**: Confirmed no recursive calls in formatting pipeline

### Key Changes
1. **_reason_with_llm**: Added explicit rules for opposites and reversed text
2. **_apply_string_subformat**: Enhanced to handle:
   - Multiple surnames (comma-separated)
   - IOC codes with case preservation
   - Better extraction logic for complex cases
3. **_format_string**: Improved case detection for acronyms and codes
4. **_format_list**: Verified clean processing without recursion

---

## CURRENT STATUS: Enhanced Agent Capabilities

### ✅ Real Data Access Implemented (PRD Section 7.2)
**The agent now fetches real data. Major data access components are complete.**

### Implementation Status

#### 1. File Processing (7.2.2) ✅ 
- **Text files**: Real file reading with content preview logging ✅
- **CSV/Excel**: pandas integration with shape/head logging ✅
- **PDF**: PyPDF2 text extraction with 500-char preview ✅
- **Images**: pytesseract OCR with extracted text logging ✅
- **Audio**: Google Cloud Speech-to-Text with MP3→WAV conversion ✅

#### 2. Web Lookup (7.2.1) ✅
- **HTTP requests**: Real web fetching with status logging
- **HTML parsing**: BeautifulSoup for content extraction
- **Error handling**: Timeouts, redirects, failures
- **Content extraction**: Title/H1 logging for verification
- **YouTube API**: Metadata fetching with title, description, tags ✅

#### 3. LLM Integration ✅
- **Ultra-concise prompts**: All LLM prompts return ONLY answer values
- **Consistent format**: "ANSWER_NOT_FOUND" for failures
- **Extraction methods**:
  - `_extract_answer_from_page_content_llm`
  - `_extract_answer_from_file_content_llm`
  - `_search_with_tavily_and_extract_llm`
  - `_reason_with_llm`

#### 4. File Attachment Handling ✅ (NEW)
- **Fixed API integration**: Questions API now passes files to agent
- **Flexible structure support**: Handles dict and list formats
- **Comprehensive logging**: Tracks file processing at each step
- **File path extraction**: Properly extracts paths from metadata

### 🎯 Enhanced Reasoning Capabilities (NEW)

#### Arithmetic Processing
- **Variable support**: Extracts and uses variables (X=5, Y=10)
- **Complex expressions**: Handles parentheses and order of operations
- **Word operations**: Interprets "times", "plus", "minus", "divided by"
- **Real calculations**: Performs actual arithmetic, not LLM guessing

#### Unit Conversions
- **Temperature**: Celsius, Fahrenheit, Kelvin
- **Distance**: meters, feet, miles, kilometers
- **Weight**: kg, lbs, grams, ounces
- **Time**: hours, minutes, seconds, days

#### Logical Deduction
- **IF-THEN-ELSE**: Evaluates conditional logic
- **Comparisons**: Greater/less than operations
- **True/False**: Boolean evaluations
- **Variable substitution**: Uses extracted variables in logic

### 📋 Key Features

#### Logging & Debugging
- **[REAL DATA]** prefix for actual data operations
- **[HANDLE_*]** prefixes for handler methods
- **[LLM_*]** prefixes for LLM operations
- Content previews for verification

#### Fallback Architecture
- Primary: LLM with ultra-concise prompts
- Secondary: Rule-based extraction/calculation
- Error handling: Graceful degradation

### 📝 Important Notes
- File attachment handling is CRITICAL and currently MISSING
- All data access components exist but file paths aren't being passed
- LLM prompts need refinement for task accuracy
- String formatting needs case preservation logic
- Comprehensive **logging** for debugging and verification

## Enhanced Answer Formatting

### Robust `_format_final_answer` Implementation
The central formatting method now includes:

1. **Pre-processing & Cleaning**:
   - `_clean_llm_response`: Removes common LLM patterns ("The answer is...", quotes, etc.)
   - `_apply_string_subformat`: Handles "first name only", "city name", etc.
   - `_extract_number_from_string`: Extracts numbers from messy text
   - `_parse_list_from_string`: Parses various list formats

2. **String Sub-formatting**:
   - **First/Last Name Extraction**: Uses LLM with fallback to regex
   - **City Name Extraction**: Removes "City of" prefixes and country suffixes
   - **Intelligent Parsing**: Handles complex name patterns and titles

3. **List Processing**:
   - **Delimiter Detection**: Comma, semicolon, bullet, numbered lists
   - **Ordering**: Alphabetical, numerical, reverse sorting
   - **Item Count Enforcement**: Ensures exact number of items
   - **Flexible Separators**: Customizable output format

4. **Number Extraction**:
   - **Currency Removal**: Strips $, €, £, USD, EUR, etc.
   - **Pattern Matching**: Finds numbers in text with decimals/negatives
   - **Comma Handling**: Removes thousands separators
   - **Fallback Logic**: Multiple extraction strategies

5. **Comprehensive Logging**:
   - `[FORMAT_FINAL]`: Main formatting flow
   - `[FORMAT_STRING]`: String-specific operations
   - `[FORMAT_LIST]`: List processing steps
   - `[FORMAT_NUMBER]`: Number extraction/formatting
   - `[STRING_SUBFORMAT]`: Sub-format applications

### Key Features
- **LLM Integration**: Uses Anthropic for complex extractions with fallbacks
- **Error Resilience**: Graceful degradation when extraction fails
- **Format Compliance**: Strict adherence to GAIA requirements
- **Debug Visibility**: Detailed logging at each transformation step

## API Integration Strategy

### Objective
Enhance BasicAgent with LLMs and specialized APIs.

### Core File
`app.py` (specifically `BasicAgent` class).

### Guiding Principle
Modify existing methods to use APIs as the primary path, with original rule-based logic as a robust fallback. MINIMIZE unnecessary code changes to non-API related parts.

### Available API Keys
From HF Secrets: ANTHROPIC_API_KEY, TAVILY_API_KEY, YOUTUBE_API_KEY, GOOGLE_API_KEY. Access via `os.getenv()`.

For local testing, use these API keys:
- ANTHROPIC_API_KEY: [REDACTED]
- GOOGLE_API_KEY: [REDACTED]
- YOUTUBE_API_KEY: [REDACTED]
- TAVILY_API_KEY: [REDACTED]
- GEMINI_API_KEY: [REDACTED]

### Primary APIs for GAIA L1
Anthropic (LLM), Tavily (Search). YouTube/Google APIs are secondary unless specific needs arise.

### Libraries
`anthropic`, `tavily-python` (ensure in `requirements.txt`). Others if YouTube/Google APIs are used (e.g., `google-api-python-client`).

### LLM Prompts
MUST instruct the LLM to return ONLY the factoid answer or structured data. NO conversational filler. Refer to GAIA formatting rules.

### Fallback
If an API call fails or an API key is missing for a primary API, the agent MUST gracefully fall back to its original rule-based logic for that step.

### Target File
All modifications: `app.py`