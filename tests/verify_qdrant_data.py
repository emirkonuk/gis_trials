#!/usr/bin/env python3
import sys

from qdrant_client import QdrantClient
from qdrant_client.http import models

COLLECTION = "hemnet_listings_v1"


def main():
    client = QdrantClient(host="127.0.0.1", port=6333, timeout=60)

    info = client.get_collection(COLLECTION)
    vectors = info.points_count
    print(f"Collection {COLLECTION}: {vectors} vectors")

    if not vectors:
        print("Collection empty; aborting")
        sys.exit(1)

    payload_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="asking_price_sek",
                range=models.Range(lte=10_000_000)
            )
        ]
    )

    hits = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=payload_filter,
        limit=5,
        with_payload=True,
    )
    points, _ = hits
    print(f"Sample query returned {len(points)} hits")
    for point in points:
        payload = point.payload or {}
        listing_id = payload.get("listing_id")
        price = payload.get("asking_price_sek")
        municipality = payload.get("municipality")
        print(f"- id={listing_id} price={price} municipality={municipality}")


if __name__ == "__main__":
    main()
