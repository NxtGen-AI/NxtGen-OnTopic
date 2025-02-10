# About
NxtGen's hub for weekly knowledge sharing, AI innovation, and collaboration, featuring LangChain, LLMs, and a range of other AI tools and techniques, combined with human insight and expertise to foster learning, growth, and innovation.

# Contents
1. [Introduction](#Introduction)
2. [Overview](#Overview)
   * [Query Answering System](#query-answering-system)
3. [Todo for Contributors](#todo-for-contributors)
4. [Adding a Submodule to the Repository](#adding-a-submodule-to-the-repository)

# Introduction
NxtGen's hub for weekly knowledge sharing, AI innovation, and collaboration is designed to bring together experts and enthusiasts in the field of artificial intelligence. By featuring LangChain, LLMs, and a range of other AI tools and techniques, combined with human insight and expertise, we aim to foster learning, growth, and innovation. This platform provides a unique opportunity for individuals to share knowledge, collaborate on projects, and stay up-to-date with the latest developments in the field.

# Overview
Our repository contains several submodules, each focused on a specific aspect of AI research and development.

## Query Answering System
A conversational AI system that takes documents and user queries, refines them, retrieves relevant documents, and generates answers using NLP techniques and LangChain framework. For more information, visit the [Query Answering System repository](https://github.com/NxtGen-AI-Public/query-answering-system).

# Todo for Contributors
- [ ] Request creation of a new submodule repository
- [ ] Clone the new submodule repository and push initial files to it, ensuring that:
  - [ ] The code has no warnings
  - [ ] The code has no linter errors (follows PEP 8 or other relevant guidelines)
  - [ ] The code follows good coding practice principles, such as:
    - Single Responsibility Principle (SRP)
    - Separation of Concerns (SoC)
    - Don't Repeat Yourself (DRY)
    - KISS (Keep it Simple, Stupid)
- [ ] Modify the contents of the [Overview](#overview) section in this README.md file to include information about the new submodule
- [ ] Ensure that each submodule has a clear and concise `README.md` file with the following sections:
  - Introduction: A brief overview of the submodule's purpose and functionality
  - Installation: Step-by-step instructions for installing and setting up the submodule
  - Usage: Examples and guidelines for using the submodule
  - System Overview: A high-level description of the submodule's architecture and components
  - Other relevant headings: Such as troubleshooting, contributing, or licensing information, as applicable to the submodule

# Adding a Submodule to the Repository
For more information on adding submodules to GitHub repositories, visit: https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/How-to-add-submodules-to-GitHub-repos

## Prerequisites
* The submodule repository must be up and running under NxtGen-AI-Public's account
* Ownership of the submodule repository is necessary for linking submodules and parent modules

## Instructions
To add a submodule to the NxtGen-OnTopic repository, follow these steps:
1. **Clone the submodule repository**: Run `git submodule add https://github.com/NxtGen-AI-Public/your-submodule-repo` (replace with your submodule repository URL)
2. **Check the status**: Run `git status`
3. **Stage changes**: Run `git add .`
4. **Commit changes**: Run `git commit -m "Add GitHub submodule"`
5. **Push changes**: Run `git push origin`