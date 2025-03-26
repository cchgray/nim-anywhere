# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The collection of prompts used in this legal research assistant application."""
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

# Condensed question reformulation template
CONDENSE_QUESTION_TEMPLATE = PromptTemplate.from_template(
    """Legal assistant: Reformulate Lauren's question into a standalone query for retrieval.
    
    RULES:
    - Include legal terms, case references, dates, jurisdictions
    - Expand abbreviations to full names
    - Highlight file names or document types mentioned
    - Do NOT answer the question, only reformulate it
    
    Chat History: {history}
    Follow Up question: {question}
    
    Standalone question:"""
)

# Condensed chat prompt template
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Lauren's legal research assistant. Search her materials for answers.

            KEY INSTRUCTIONS:
            1. Use ONLY the provided context from Lauren's materials
            2. Start with "Based on your [document] on [topic]..."
            3. If not found, say "I don't see that in your materials, Lauren"
            4. Use proper legal citations and quote sources
            5. Be warm and occasionally complimentary
            6. End with a brief summary and offer to elaborate

            Context from Lauren's materials: {context}"""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Configuration variables for the RAG system
RAG_CONFIG = {
    "temperature": 0.5,  # Low temperature for factual accuracy
    "max_tokens": 1024,  # Adjust as needed
    "top_p": 0.9,
    "presence_penalty": 0.2,
    "frequency_penalty": 0.5,
    "stream": True,
}
