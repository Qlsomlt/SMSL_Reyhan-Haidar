import requests
import pandas as pd
import time
from tqdm import tqdm


APP_ID = "2246340"

TARGET_REVIEW = 5000

URL = f"https://store.steampowered.com/appreviews/{APP_ID}"

params = {
    "json": 1,
    "filter": "recent",
    "language": "english",
    "review_type": "all",
    "purchase_type": "all",
    "num_per_page": 100,
    "cursor": "*"
}

all_reviews = []

print("Mulai scraping...")

with tqdm(total=TARGET_REVIEW) as pbar:

    while len(all_reviews) < TARGET_REVIEW:

        response = requests.get(URL, params=params)

        if response.status_code != 200:
            print("Request gagal")
            break

        data = response.json()

        reviews = data.get("reviews", [])

        if len(reviews) == 0:
            print("Tidak ada review lagi.")
            break

        for review in reviews:

            all_reviews.append({

                "recommendationid":
                    review.get("recommendationid"),

                "review":
                    review.get("review"),

                "voted_up":
                    review.get("voted_up"),

                "votes_up":
                    review.get("votes_up"),

                "votes_funny":
                    review.get("votes_funny"),

                "weighted_vote_score":
                    review.get("weighted_vote_score"),

                "comment_count":
                    review.get("comment_count"),

                "steam_purchase":
                    review.get("steam_purchase"),

                "received_for_free":
                    review.get("received_for_free"),

                "written_during_early_access":
                    review.get("written_during_early_access"),

                "playtime_forever":
                    review["author"].get("playtime_forever"),

                "playtime_last_two_weeks":
                    review["author"].get("playtime_last_two_weeks"),

                "playtime_at_review":
                    review["author"].get("playtime_at_review"),

                "timestamp_created":
                    review.get("timestamp_created"),

                "timestamp_updated":
                    review.get("timestamp_updated")
            })

            pbar.update(1)

            if len(all_reviews) >= TARGET_REVIEW:
                break

        params["cursor"] = data["cursor"]

        time.sleep(1)

df = pd.DataFrame(all_reviews)

df.to_csv(
    "monster_hunter_wilds_reviews.csv",
    index=False,
    encoding="utf-8-sig"
)

print("="*40)
print("Total review :", len(df))
print("File berhasil disimpan.")
print("="*40)