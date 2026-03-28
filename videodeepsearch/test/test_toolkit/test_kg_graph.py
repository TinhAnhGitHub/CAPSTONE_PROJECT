import asyncio
from loguru import logger

from arango.client import ArangoClient
from videodeepsearch.clients.inference import MMBertClient, MMBertConfig
from videodeepsearch.toolkit.kg_retrieval import KGSearchToolkit
from videodeepsearch.clients.storage.arangodb.index_manager import ArangoIndexManager

CONFIG = {
    "arango_host": "http://localhost:8529",
    "arango_database": "video_kg",
    "arango_username": "root",
    "arango_password": "",
    "mmbert_url": "http://localhost:8009",
    "video_ids":[
    "0e64f1c0da591ca67f07b7f9",
    "3636d10a2ad4787733c9700d",
    "946330031ead69b21354d038",
    "9b17f473300a5436f0a053be",
    "c510fac771767405c891bf64",
    "c98019fd17ff4420ea47eee7"
],
    "top_k": 15,
    "queries": {
        "entities_semantic": "SIMEX",
        "events": "The scene contains SIMEX stock exchange displaying a 40000 dollar  graphic.",
        "micro_events": "Reichelt and Eiffel tower. ",
        "communities": "A community descbibes nick leason and his trading stock market",
        "bm25": "machine learning neural network",
        "multi_granularity": "The scene contains SIMEX stock exchange displaying a 40000 dollar  graphic.",
        "triple_hybrid": "The scene contains SIMEX stock exchange displaying a 40000 dollar  graphic.",
        "triple_hybrid_all": "people collaborating on a project",
        "rag": "The scene contains SIMEX stock exchange displaying a 40000 dollar  graphic.",
    },
}


def _section(title: str) -> None:
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def _print_result(result, title: str = "") -> None:
    """Pretty print a ToolResult."""
    if title:
        print(f"\n--- {title} ---")
    if hasattr(result, "content"):
        print(result.content)
    else:
        print(result)


async def test_index_creation(db) -> dict:
    """Test ArangoDB index creation."""
    _section("Test: Index Creation")

    arango_index_manager = ArangoIndexManager(db=db)
    result = await arango_index_manager.ensure_all_indexes()

    print("\nIndex creation results:")
    for key, success in result.items():
        status = "✓" if success else "✗"
        print(f"  {status} {key}")

    return result


async def test_semantic_entity_search(toolkit: KGSearchToolkit, cfg: dict) -> None:
    """Test semantic entity search."""
    _section("Test: Semantic Entity Search")

    query = cfg["queries"]["entities_semantic"]
    print(f"Query: '{query}'")

    result = await toolkit.search_entities_semantic.entrypoint(
        toolkit,
        query=query,
        top_k=cfg["top_k"],
        user_id='tinhanhuser'
        # video_ids=cfg["video_ids"],
    )
    _print_result(result, "Results")


async def test_event_search(toolkit: KGSearchToolkit, cfg: dict) -> None:
    """Test event search."""
    _section("Test: Event Search")

    query = cfg["queries"]["events"]
    print(f"Query: '{query}'")

    result = await toolkit.search_events.entrypoint(
        toolkit,
        query=query,
        top_k=cfg["top_k"],
        video_ids=cfg["video_ids"],
    )
    _print_result(result, "Results")


async def test_micro_event_search(toolkit: KGSearchToolkit, cfg: dict) -> None:
    """Test micro-event search."""
    _section("Test: Micro-Event Search")

    query = cfg["queries"]["micro_events"]
    print(f"Query: '{query}'")

    result = await toolkit.search_micro_events.entrypoint(
        toolkit,
        query=query,
        top_k=cfg["top_k"],
        video_ids=cfg["video_ids"],
    )
    _print_result(result, "Results")


async def test_community_search(toolkit: KGSearchToolkit, cfg: dict) -> None:
    """Test community search."""
    _section("Test: Community Search")

    query = cfg["queries"]["communities"]
    print(f"Query: '{query}'")

    result = await toolkit.search_communities.entrypoint(
        toolkit,
        query=query,
        top_k=cfg["top_k"],
        video_ids=cfg["video_ids"],
    )
    _print_result(result, "Results")

async def test_multi_granularity_search(toolkit: KGSearchToolkit, cfg: dict) -> None:
    """Test multi-granularity search across entity, event, and micro-event levels."""
    _section("Test: Multi-Granularity Search")

    query = cfg["queries"]["multi_granularity"]
    print(f"Query: '{query}'")

    result = await toolkit.multi_granularity_search.entrypoint(
        toolkit,
        query=query,
        top_k=cfg["top_k"],
        video_ids=cfg["video_ids"],
    )
    _print_result(result, "Results")

async def test_triple_hybrid_search(toolkit: KGSearchToolkit, cfg: dict) -> None:
    """Test triple-hybrid search (BM25 + semantic + graph structure)."""
    _section("Test: Triple-Hybrid Search")

    query = cfg["queries"]["triple_hybrid"]
    print(f"Query: '{query}'")

    result = await toolkit.triple_hybrid_search.entrypoint(
        toolkit,
        query=query,
        top_k=15,
        video_ids=cfg["video_ids"],
        search_all_collections=True
    )
    _print_result(result, "Results (Entities)")


async def test_rag_retrieval(toolkit: KGSearchToolkit, cfg: dict) -> None:
    """Test RAG-style retrieval combining all retrieval methods."""
    _section("Test: RAG Retrieval")

    query = cfg["queries"]["rag"]
    print(f"Query: '{query}'")

    result = await toolkit.retrieve_for_rag.entrypoint(
        toolkit,
        query=query,
        video_ids=cfg["video_ids"],
        top_k_entities=5,
        top_k_events=3,
        top_k_communities=2,
    )
    _print_result(result, "RAG Context")


async def test_graph_traversal(toolkit: KGSearchToolkit, cfg: dict) -> None:
    """Test graph traversal from a seed entity."""
    _section("Test: Graph Traversal")

    # First, search for an entity to use as seed
    query = cfg["queries"]["entities_semantic"]
    _ = await toolkit.search_entities_semantic.entrypoint(
        toolkit,
        query=query,
        top_k=1,
        video_ids=cfg["video_ids"],
    )

    # Extract entity key from results
    if toolkit._result_store:
        handle_id = list(toolkit._result_store.keys())[0]
        results = toolkit._result_store[handle_id]
        if results:
            entity_key = results[0].get("_key", "")
            if entity_key:
                print(f"Starting traversal from entity: {entity_key}")

                result = toolkit.traverse_from_entity.entrypoint(
                    toolkit,
                    entity_key=entity_key,
                    max_depth=2,
                )
                _print_result(result, "Traversal Results")
                return

    print("No entities found for traversal test")


async def main():
    cfg = CONFIG

    _section("Initializing Clients")

    client = ArangoClient(hosts=cfg["arango_host"])  # type: ignore
    db = client.db(
        cfg["arango_database"],  # type: ignore
        username=cfg["arango_username"],  # type: ignore
        password=cfg["arango_password"],  # type: ignore
    )
    logger.info(f"Connected to ArangoDB: {cfg['arango_database']}")

    mmbert_client = MMBertClient(MMBertConfig(base_url=str(cfg["mmbert_url"])))
    logger.info(f"Connected to MMBert: {cfg['mmbert_url']}")

    toolkit = KGSearchToolkit(
        arango_db=db,
        mmbert_client=mmbert_client,
    )
    logger.info("Initialized KGSearchToolkit")

    
    try:
        # # 1. Index creation
        # await test_index_creation(db)

        # # 2. Semantic entity search
        await test_semantic_entity_search(toolkit, cfg)

        # # # 3. Event search
        # await test_event_search(toolkit, cfg)

        # # # 4. Micro-event search
        # await test_micro_event_search(toolkit, cfg)

        # # # 5. Community search
        # await test_community_search(toolkit, cfg)

        # # 6. Multi-granularity search
        # await test_multi_granularity_search(toolkit, cfg)

        # # # 7. Triple-hybrid search
        # await test_triple_hybrid_search(toolkit, cfg)

        # # # 8. RAG retrieval
        # await test_rag_retrieval(toolkit, cfg)

        # # # 9. Graph traversal
        # await test_graph_traversal(toolkit, cfg)

        _section("All Tests Complete")

    finally:
        await mmbert_client.close()
        client.close()
        logger.info("Closed connections")


if __name__ == "__main__":
    asyncio.run(main())