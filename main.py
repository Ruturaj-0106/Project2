import pandas as pd
import numpy as np
from textblob import TextBlob

# Load datasets
books = pd.read_csv("books.csv")
ratings = pd.read_csv("ratings.csv")
book_tags = pd.read_csv("book_tags.csv")
to_read = pd.read_csv("toread.csv")

# Q1: How many unique books are present in the dataset?
unique_books = books['book_id'].nunique()
print(f"Q1: No of Unique books are {unique_books}")

# Q2: How many unique users are present in the dataset?
unique_users = ratings['user_id'].nunique()
print(f"Q2: No of Unique users are {unique_users}")

# Q3: How many books do not have an original title?
missing_titles = books['original_title'].isnull().sum()
print(f"Q3: Books without an original title: {missing_titles}")

# Q4: Which book has the maximum number of ratings?
ratings_count = ratings.groupby('book_id').size()
max_rated_book_id = ratings_count.idxmax()

# Check if the book_id exists in the books dataframe
if max_rated_book_id in books['book_id'].values:
    max_rated_book = books.loc[books['book_id'] == max_rated_book_id, 'original_title'].values[0]
    print(f"Q4: Book with max ratings: {max_rated_book}")
else:
    print(f"Q4: Book with ID {max_rated_book_id} not found in books dataset")


# Q5: Most frequently used tag_id
most_frequent_tag = book_tags['tag_id'].value_counts().idxmax()
print(f"Q5: Most frequent tag_id is {most_frequent_tag}")

# Q6: Book with most tag counts
book_tags_grouped = book_tags.groupby('goodreads_book_id')['count'].sum()
book_with_most_tags_id = book_tags_grouped.idxmax()

# Check if the book_with_most_tags_id exists in the books dataframe
if book_with_most_tags_id in books['goodreads_book_id'].values:
    book_with_most_tags = books.loc[books['goodreads_book_id'] == book_with_most_tags_id, 'original_title'].values[0]
    print(f"Q6: Book with most tags: {book_with_most_tags}")
else:
    print(f"Q6: Book with Goodreads Book ID {book_with_most_tags_id} not found in books dataset")

# Q7: Book marked as to-read by most users
# We group by book_id to get the most frequently marked "to-read" book
most_to_read_book_id = to_read['book_id'].value_counts().idxmax()

# Check if the book_id exists in the books dataframe
if most_to_read_book_id in books['book_id'].values:
    most_to_read_book = books.loc[books['book_id'] == most_to_read_book_id, 'original_title'].values[0]
    print(f"Q7: Most to-read book: {most_to_read_book}")
else:
    print(f"Q7: Book with ID {most_to_read_book_id} not found in books dataset")

# Q8: Least used tag_id
least_used_tag = book_tags['tag_id'].value_counts().idxmin()
print(f"Q8: Least used tag_id is {least_used_tag}")

# Q9: Book with minimum average rating
# We calculate average rating for each book from the ratings dataset
avg_ratings = ratings.groupby('book_id')['rating'].mean()
min_avg_rating_book_id = avg_ratings.idxmin()

# Check if the book_id exists in the books dataframe
if min_avg_rating_book_id in books['book_id'].values:
    min_avg_rating_book = books.loc[books['book_id'] == min_avg_rating_book_id, 'original_title'].values[0]
    print(f"Q9: Book with minimum avg rating: {min_avg_rating_book}")
else:
    print(f"Q9: Book with ID {min_avg_rating_book_id} not found in books dataset")

# Q10: Book with least tag count
# Grouping by book_id to get the total count of tags per book
book_tag_counts_least = book_tags.groupby('goodreads_book_id')['count'].sum()
least_tagged_book_id = book_tag_counts_least.idxmin()

# Check if the least_tagged_book_id exists in the books dataframe
if least_tagged_book_id in books['goodreads_book_id'].values:
    least_tagged_book = books.loc[books['goodreads_book_id'] == least_tagged_book_id, 'original_title'].values[0]
    print(f"Q10: Book with least tags: {least_tagged_book}")
else:
    print(f"Q10: Book with Goodreads Book ID {least_tagged_book_id} not found in books dataset")

# Q11: Total unique tags
unique_tags = book_tags['tag_id'].nunique()
print(f"Q11: Total unique tags: {unique_tags}")

# Q12: Average rating of all books
# Average rating across all books
average_rating = avg_ratings.mean()
print(f"Q12: Average rating of all books: {average_rating:.2f}")

# Q13: Number of books published in 2000
books_published_in_2000 = books[books['original_publication_year'] == 2000].shape[0]
print(f"Q13: Books published in 2000: {books_published_in_2000}")

# Q14: Sentiment analysis of book titles
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

books["polarity"] = books["original_title"].fillna("").apply(get_polarity)
books["sentiment"] = books["polarity"].apply(
    lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral")
)
negative_sentiment_count = (books["sentiment"] == "negative").sum()
print(f"Q14: Number of negative sentiment titles: {negative_sentiment_count}")
