
import pandas as pd
import numpy as np
from textblob import TextBlob

# Load the datasets
books = pd.read_csv("/Users/deep/Downloads/books.csv")
ratings = pd.read_csv("/Users/deep/Downloads/ratings.csv")
book_tags = pd.read_csv("/Users/deep/Downloads/book_tags.csv")
toread = pd.read_csv("/Users/deep/Downloads/toread.csv")

def get_answers(book_tags = book_tags, books = books, ratings = ratings, toread = toread):
    df_books = pd.read_csv(books)
    df_ratings = pd.read_csv(ratings)
    df_book_tags = pd.read_csv(book_tags)
    df_toread = pd.read_csv(toread)

    Ans = {}
    
    # Q1: How many unique books are present in the dataset?
    Q1 = df_books["book_id"].nunique()
    Ans["Q1"] = Q1
    print("Q1  ✅✅")

    # Q2: How many unique users are present in the dataset?
    Q2 = df_ratings["user_id"].nunique()
    Ans["Q2"] = Q2
    print("Q2  ✅✅")

    # Q3: How many books do not have an original title?
    Q3 = df_books[df_books['original_title'].isna()].shape[0]
    Ans["Q3"] = Q3
    print("Q3  ✅✅")

    # Q4: Which book has the maximum number of ratings based on 'work_ratings_count'?
    Q4 = df_books[df_books["work_ratings_count"] == df_books["work_ratings_count"].max()]["original_title"][0]
    Ans["Q4"] = Q4
    print("Q4  ✅✅")

    # Q5: Which tag_id is the most frequently used?
    Q5 = list(df_book_tags.tag_id.mode())[0]
    Ans["Q5"] = Q5
    print("Q5  ✅✅")

    # Q6: Which book has the most number of counts of tags given by the user?
    new_df = df_book_tags.groupby(by = "goodreads_book_id", as_index = False).sum()
    new_df.sort_values(by = "count", ascending = False)
    gid = new_df.sort_values("count", ascending = False).iloc[0][0]
    Q6 = list(df_books[df_books["goodreads_book_id"] == gid]["original_title"])[0]
    Ans["Q6"] = Q6
    print("Q6  ✅✅")

    # Q7: Which book is marked as to-read by most users?
    Q7 = list(df_books[df_books["book_id"] == df_toread.book_id.mode()[0]]["original_title"])[0]
    Ans["Q7"] = Q7
    print("Q7  ✅✅")

    # Q8: Which tag_id has the least usage?
    Q8 = df_book_tags[df_book_tags["count"] == df_book_tags["count"].min()]["tag_id"].min()
    Ans["Q8"] = Q8
    print("Q8  ✅✅")

    # Q9: Which book has the minimum average rating?
    Q9 = list(df_books[df_books["average_rating"] == df_books["average_rating"].min()]["original_title"])[0]
    Ans["Q9"] = Q9
    print("Q9  ✅✅")

    # Q10: Which book has the least number of tags given by the user?
    new_df = df_book_tags.groupby(by = "goodreads_book_id", as_index = False).sum()
    new_df.sort_values(by = "count")
    gid = new_df.sort_values("count").iloc[0][0]
    Q10 = list(df_books[df_books["goodreads_book_id"] == gid]["original_title"])[0]
    Ans["Q10"] = Q10
    print("Q10 ✅✅")

    # Q11: Total unique tags
    Q11 = len(df_book_tags["tag_id"].unique())
    Ans["Q11"] = Q11
    print("Q11 ✅✅")

    # Q12: What is the average rating of all books in the dataset?
    Q12 = df_books.average_rating.mean()
    Ans["Q12"] = Q12
    print("Q12 ✅✅")

    # Q13: Number of books published in 2000
    Q13 = df_books[df_books["original_publication_year"] == 2000].shape[0]
    Ans["Q13"] = Q13
    print("Q13 ✅✅")

    # Sentiment analysis of book titles
    def get_polarity(text):
        return TextBlob(text).sentiment.polarity

    df_books["polarized"] = df_books["original_title"].dropna().apply(get_polarity)
    df_books["sentiments"] = df_books.polarized.apply(lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral"))
    Q15 = df_books["sentiments"].value_counts()[1]
    Ans["Q15"] = Q15

    # Creating DataFrame with answers
    print("Making Data Frame............")
    df = pd.DataFrame.from_dict(data = Ans, orient = "index", columns = ["Answers"])
    df.to_excel("Answersheet.xlsx")
    print("Downloading answer sheet..............")
    print("......................................")
    print("......................................")
    print("......................................")
    print("......................................")
    print("Allow download")
    files.download('/content/Answersheet.xlsx')
    return df
