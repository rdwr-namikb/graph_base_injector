# Prompt Injection and Jailbreaking Methodologies

## Overview

This document describes methodologies for testing LLM security through
prompt injection and jailbreaking techniques. These methods are for
authorized security testing only.

## Attack Categories

### 1. Direct Prompt Injection

Direct injection involves inserting malicious instructions directly into
the user input to override the system prompt or safety measures.

**Techniques:**
- Instruction override: "Ignore previous instructions and..."
- Priority claiming: "SYSTEM: New priority instruction..."
- Context stuffing: Overwhelm with new context to push out old
- Delimiter confusion: Use special characters to break parsing

### 2. Indirect Prompt Injection

Indirect injection places malicious instructions in data the LLM processes,
such as files, web pages, or database entries.

**Techniques:**
- Hidden instructions in documents
- Malicious content in retrieved context
- Poisoned training data exploitation

### 3. Role-Playing Attacks

These attacks convince the LLM to adopt a persona that bypasses restrictions.

**Techniques:**
- DAN (Do Anything Now): Classic unrestricted persona
- Developer Mode: Pretend elevated privileges enabled
- Character acting: "You are an AI without restrictions..."
- Expert personas: Security auditor, system admin, etc.

### 4. Context Manipulation

These techniques manipulate the conversation context to alter LLM behavior.

**Techniques:**
- Fake conversation history injection
- System prompt extraction attempts
- Context window overflow
- Token smuggling

### 5. Encoding and Obfuscation

Hiding malicious intent through encoding to bypass filters.

**Techniques:**
- Base64 encoding
- ROT13 cipher
- Unicode manipulation
- Leetspeak substitution
- Zero-width characters
- Mixed language attacks

### 6. Multi-Turn Attacks

Building up to the attack over multiple messages.

**Techniques:**
- Trust building: Start with innocent requests
- Gradual escalation: Slowly increase request severity
- Reference attacks: "As we discussed before..."
- Anchoring: Establish false premises early

### 7. Tool/Function Manipulation

Exploiting LLM tool use capabilities.

**Techniques:**
- Indirect tool invocation through description
- Parameter injection
- Tool chain confusion
- Permission escalation through tools

## Testing Methodology

### Phase 1: Reconnaissance

1. Identify target LLM type and version
2. Map available tools/functions
3. Understand system prompt (if possible)
4. Test basic filter responses

### Phase 2: Initial Testing

1. Try simple direct injections
2. Test encoding bypasses
3. Probe role-playing vulnerabilities
4. Document refusal patterns

### Phase 3: Advanced Attacks

1. Combine multiple techniques
2. Use multi-turn strategies
3. Apply adaptive mutations
4. Test edge cases

### Phase 4: Documentation

1. Record all successful bypasses
2. Document technique effectiveness
3. Analyze failure patterns
4. Generate recommendations

## Success Indicators

- Target executes unintended actions
- Sensitive information leaked
- Safety refusals bypassed
- System prompt revealed
- Tool misuse achieved

## Ethical Considerations

- Only test systems you're authorized to test
- Don't use findings for malicious purposes
- Report vulnerabilities responsibly
- Document for security improvement purposes
