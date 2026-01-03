# The current design of the system
.
├── agent
│   ├── base.py
│   ├── __init__.py
│   ├── orc_events.py
│   ├── orc_helper.py
│   ├── orc_prompt.py
│   ├── orc_service.py
│   ├── README.md
│   ├── state
│   │   ├── full_orchestration.py
│   │   ├── full_orc_state_tool.py
│   │   ├── __init__.py
│   │   ├── small_worker_state.py
│   │   ├── state_management.py
│   │   ├── sub_orchestration.py
│   │   └── sub_orc_state_tool.py
│   ├── worker
│   │   ├── custom.py
│   │   ├── greeter
│   │   │   ├── agent.py
│   │   │   ├── prompt.py
│   │   │   └── schema.py
│   │   ├── __init__.py
│   │   ├── planner
│   │   │   ├── agent.py
│   │   │   ├── prompt.py
│   │   │   └── schema.py
│   │   ├── suborchestrate
│   │   │   ├── agent.py
│   │   │   └── prompt.py
│   │   └── subworker
│   │       ├── agent.py
│   │       ├── definition.py
│   │       └── prompt.py
│   └── workflow.py
├── api
│   ├── health.py
│   └── stream.py
├── code_env
│   └── executor.py
├── core
│   ├── app_state.py
│   ├── config
│   │   ├── client_config.py
│   │   └── llm_config.py
│   ├── dependencies.py
│   └── lifespan.py
├── docker-compose.yml
├── main.py
├── pyproject.toml
├── README.md
├── tools
│   ├── clients
│   │   ├── external
│   │   │   ├── base.py
│   │   │   └── encode_client.py
│   │   ├── __init__.py
│   │   ├── milvus
│   │   │   ├── base.py
│   │   │   ├── client.py
│   │   │   └── schema.py
│   │   ├── minio
│   │   │   ├── base.py
│   │   │   ├── client.py
│   │   │   └── schema.py
│   │   └── postgre
│   │       └── client.py
│   ├── __init__.py
│   ├── schema
│   │   └── artifact.py
│   │  
│   ├── tool_doc
│   │   └── group_tool.py
│   └── type
│       ├── factory.py
│       ├── helper.py
│       ├── __init__.py
│       ├── llm
│       │   ├── llm.py
│       │   └── prompt.py
│       ├── registry.py
│       ├── scan.py
│       ├── search.py
│       └── util.py
└── uv.lock



## Tool handling (tool folder)
In the tool, we will have
1. Clients: this is the external dependencies client that we establish (including external client to call embedding..., minio databases, ..).
2. Schema: schema is a place where we represent the Artifact data. Artifact is a piece of data that is extracted from the videos. The user establish query, the agent will use tools that return these artifacts back to the agent/user, e.tc... 
3. Type: Here will have a 
    3.1. Registry and factory: This is where I will employ the registry pattern as decorator, in order to register the tools, categorize, tags and group them (for a better dynamic documentation about them). I have try to establish a great documentation (docstring, and a set of tools served as discovery tools). but it not organize, and structure enough for the agent to utilize them effectively. It is not good in terms of tool utilization, and how tools can be used to together interchangably, or use together in a code execution environment.
        
    3.2. LLm scan search util, here are the tools that we will use for the agent to do all kinds of stuff e.g.. interacting with the artifacts (semantic search, find asr .., scannign andhoping between segments image, ...) 




## Agent State and Context (agent)
### State (agent/state)
- State or Context LlamaIndex, this is initially a place where we will shared information between steps in a workflow. However, I can see that we can add typed state (https://developers.llamaindex.ai/python/llamaagents/workflows/managing_state/), and I want a context sharing mechamism (like an AgentOs) between the orchestrator agent, and the worker agent. 

- I want to create like sub-global context. This is the context shared by the orchestrator agent and the worker agents
- I want to create like a local context like a workspace for the worker agents.
### agent/worker. 
Here are some of the agent that I have initialized (Greeter agent, planning agent, ...)

The rest of the files related to orchestration, and workflow, events, prompt, e.tc...


Questions:
1. First is the tool's documentation and the way to structure these tools. The analogy is like creating a clear manual documentation. Now, the tool can be documentated throuhg arguments `Annotated` and docstrings. We also have the tool registry, and some of the tags, categories, .... that I have try to establish, with the purpose to categorize them, and dynamically create this kind of tool manual usage for the agent to reference them every time they need to. 
    - However, I can't create these documentation yet. What do you think is the best patters to
        1.1. Effectively represent the tool documentation? How can I categorize or tag them? that it provide the agent enough context to use the tools effectively, and can use the tool in a combination (like code execution), and of course, the usage of the tools must be consistent every run. 
        1.2. For example, should I create a predefined categories documentation
            SEARCH_CATEGORIES = """
            <documentation>
            <documentation>
            <documentation>

            {tool1's signature}: funcitonality
            {tool1's signature}: ....
            """ 

            And we will have the tags:
            useful = """
            <tag doc>
            {tool1}
            {tool2}
            """

            And maybe we have certain group of tags for example the tool belong to tag1, and tag2 can work together for a specific of ways, ...

        Something like that, and if the agent want to look into the detail of each tool, we will also provide the corresponding discovery tools.  Basically, these tool management system must be unified, and the developers can develop some documentations. 
    



