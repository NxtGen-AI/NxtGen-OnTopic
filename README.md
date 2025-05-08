# About 📚
NxtGen's hub for weekly knowledge sharing, AI innovation, and collaboration, featuring LangChain, LLMs, and a range of other AI tools and techniques, combined with human insight and expertise to foster learning, growth, and innovation.

# Contents 📝
1. [Introduction 📚](#introduction)
2. [Overview 🤖](#overview)
   * [Query Answering System 💬](#query-answering-system)
   * [Self-RAG 💬]
3. [TODO for Contributors 📝](#todo-for-contributors)
4. [Adding a Submodule to the Repository 📁](#adding-a-submodule-to-the-repository)
5. [Modifying Submodule's Tracked Branch](#modifying-submodules-tracked-branch)
6. [Updating Submodule to Track Latest Commit](#updating-submodule-to-track-latest-commit)

# Introduction 📚
NxtGen's hub for weekly knowledge sharing, AI innovation, and collaboration is designed to bring together experts and enthusiasts in the field of artificial intelligence. By featuring LangChain, LLMs, and a range of other AI tools and techniques, combined with human insight and expertise to foster learning, growth, and innovation.

# Overview 🤖
Our repository contains several submodules, each focused on a specific aspect of AI research and development.

## Query Answering System 💬
A conversational AI system that takes documents and user queries, refines them, retrieves relevant documents, and generates answers using NLP techniques and LangChain framework. For more information, visit the [Query Answering System repository](https://github.com/NxtGen-AI-Public/query-answering-system).

## Self-RAG 💬
Self-RAG (Self-Reflective Retrieval-Augmented Generation) enhances the performance of Large Language Models (LLMs) by enabling them to self-reflect on their responses and adapt their behavior accordingly. It achieves this through a combination of on-demand retrieval of external knowledge, generation of responses, and self-critique using reflection tokens. This allows the LLM to improve its factuality, verifiability, and overall quality of generation. For more information, visit the [Query Answering System repository](https://github.com/NxtGen-AI/self-rag).

## Agent-to-Agent (A2A) Protocol
The Agent-to-Agent (A2A) Protocol is a communication framework that enables secure, decentralized messaging between autonomous agents. Commonly used in decentralized identity (DID) ecosystems like Hyperledger Aries, A2A facilitates the exchange of verifiable credentials, proofs, and DIDComm messages between agents without centralized intermediaries. This repository provides a foundational implementation or reference for A2A message structures, routing, encryption, and transport layers involved in DID-based agent communications.[A2A Protocol repository](https://github.com/NxtGen-AI/a2a-protocol.git).

# TODO for Contributors 📝
- [ ] Request creation of a new submodule repository
- [ ] Clone the new submodule repository and push initial files to it, ensuring that:
  - [ ] The code has no warnings
  - [ ] The code has no linter errors (follows PEP 8 or other relevant guidelines)
  - [ ] The code is thoroughly tested before being pushed
  - [ ] The code follows good coding practice principles, such as:
    - Single Responsibility Principle (SRP)
    - Separation of Concerns (SoC)
    - Don't Repeat Yourself (DRY)
    - KISS (Keep it Simple, Stupid)
- [ ] Modify the contents of the [Overview 🤖](#overview) section in this README.md file to include information about the new submodule
- [ ] Ensure that each submodule (a.k.a subrepo) has a clear and concise `README.md` file with the following sections:
  - Introduction: A brief overview of the submodule's purpose and functionality
    - **Include a link to the parent module/repo (https://github.com/NxtGen-AI-Public/NxtGen-OnTopic) to provide context and facilitate navigation**
  - Installation: Step-by-step instructions for installing and setting up the submodule
  - Usage: Examples and guidelines for using the submodule
  - System Overview: A high-level description of the submodule's architecture and components
  - Other relevant headings: Such as troubleshooting, contributing, or licensing information, as applicable to the submodule

# Adding a Submodule to the Repository 📁
For more information on adding submodules to GitHub repositories, visit: https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/How-to-add-submodules-to-GitHub-repos

## Prerequisites 📝
* The submodule repository must be up and running under NxtGen-AI-Public's account
* Ownership of the submodule repository is necessary for linking submodules and parent modules

## Instructions 📚
To add a submodule to the NxtGen-OnTopic repository, follow these steps:
1. **Create a GitHub repository**: Create a new GitHub repository under the `NxtGen-AI-Public` account. This is necessary because submodules must belong to the same owner as the parent repository.
2. **Clone the submodule repository**: Run `git submodule add https://github.com/NxtGen-AI-Public/your-submodule-repo` (replace with your submodule repository URL)
3. **Check the status**: Run `git status`
4. **Stage changes**: Run `git add .`
5. **Commit changes**: Run `git commit -m "Add GitHub submodule"`
6. **Push changes**: Run `git push origin`

# Modifying Submodule Tracked Branch 📁
Git submodule objects are special kinds of Git objects, and **they hold the SHA information for a specific commit**, **they do not track the branch and update as and when any commit is pushed into the branch**, they have the latest changes of the latest commit at the time of adding the submodule. For more information on updating submodules to track different branches in GitHub repositories, visit: https://stackoverflow.com/a/18797720.

## Prerequisites 📝
* The submodule repository must be up and running under NxtGen-AI-Public's account
* Ownership of the submodule repository is necessary for linking submodules and parent modules
* A submodule must already exist in the repository

## Instructions 📚
To update a submodule to track a different branch, follow these steps:
1. **Update the submodule's configuration to track the new branch**: 
   - Manually edit the `.gitmodules` file in the superproject directory: add or modify the line `branch = main` (replace with your desired branch) under the `[submodule "path/to/your/submodule"]` section.
   - Alternatively, run `git config -f .gitmodules submodule.path/to/your/submodule.branch main` from the superproject root
2. **Commit the changes in the `.gitmodules` file**: Run `git add .gitmodules` and then `git commit -m "Updated submodule to track main branch"`
3. **Push changes**: Run `git push origin`

# Updating a Git Submodule to Track the Latest Commit

Git submodules **do not automatically track branches**. Instead, they **point to a specific commit** in the submodule repository. If you want your parent repo to use the latest commit from a submodule’s branch (e.g., `main`), follow these steps.

📖 For a deeper explanation, see: https://stackoverflow.com/a/18797720

---

## ✅ Prerequisites

- The submodule repository must exist and be accessible
- The submodule must already be added to the parent repository
- You should have push access to both the parent and submodule repositories

---

## 🔄 Steps to Update a Submodule to the Latest Commit

### 1. Fetch the latest commit in the submodule

From the root of your parent repository:

    cd <submodule-folder>                  # Navigate into the submodule
    git checkout <branch-name>            # Checkout the correct branch (e.g., main)
    git pull origin <branch-name>         # Pull the latest commit
    cd ..                                 # Return to the parent repository

> Replace `<submodule-folder>` with the name of your submodule and `<branch-name>` with the branch you want to track (usually `main` or `master`).

---

### 2. Stage the updated submodule commit reference

    git add <submodule-folder>

This stages the updated submodule commit (the pointer in the parent repo).

---

### 3. Commit the change

    git commit -m "Update <submodule-folder> to latest commit"

---

### 4. Push the changes to the remote

    git push origin <branch-name>

---
