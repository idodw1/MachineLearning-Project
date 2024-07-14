import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import pandas as pd
import numpy as np
import string
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression


''' Raw Data Edits Functions '''


# Raw data transformations
def raw_data_trans(df):
    df_cop = df.copy()
    # Apply the function to create a new column with domain names
    df_cop['domain'] = df_cop['email'].apply(extract_domain)
    df_cop['num_of_follows'] = df_cop['date_of_new_follow'].apply(len)
    df_cop['num_of_followers'] = df_cop['date_of_new_follower'].apply(len)
    df_cop['num_of_previous_messages'] = df_cop['previous_messages_dates'].apply(len)
    # Extract the year from the datetime column
    df_cop['year'] = pd.to_datetime(df_cop['account_creation_date']).dt.year
    # Filter for the years you're interested in (2013, 2014, 2015)
    years_of_interest = [2013, 2014, 2015]
    df_cop = df_cop[df_cop['year'].isin(years_of_interest)]
    # Extract the quarter from the datetime column
    df_cop['quarter'] = pd.to_datetime(df_cop['message_date']).dt.quarter
    # Count the number of words in each row of the 'text' column
    df_cop['word_count'] = df_cop['text'].apply(lambda x: len(x.split()))
    return df_cop


# Function to extract domain names
def extract_domain(email):
    if pd.isnull(email):
        return None
    else:
        return email.split('@')[1].split('.')[0]


''' Graph plots Functions '''


# Bar plot of regular variable
def variable_plot(variable_data, total_data_points, header, x_label, y_label, figure_index):
    plt.figure(figure_index, figsize=(10, 6))
    for i, count in enumerate(variable_data.values):
        percentage = '{:.1f}%'.format(100 * count / total_data_points)
        plt.text(i, count, percentage, ha='center', va='bottom', fontsize=10)
    sns.barplot(x=variable_data.index, y=variable_data.values, palette="viridis")
    plt.title(header)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels


# Bar plot of categorical variable
def multi_categories_plot(variable_data, header, x_label, y_label, figure_index):
    plt.figure(figure_index, figsize=(10, 6))
    sns.barplot(x=variable_data.index, y=variable_data.values, palette="viridis")
    plt.title(header)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels


# Generate plots to all the variables in the raw data set
def raw_data_plots(df):
    # Platform plot
    variable_plot(df['platform'].value_counts(), len(df['platform']),
                  'Platform Distribution', 'Platform', 'Count', 1)
    # Embedded content plot
    variable_plot(df['embedded_content'].value_counts(), len(df['embedded_content']),
                  'Embedded Distribution', 'Embedded Content', 'Count', 2)
    # Blue Tick plot
    variable_plot(df['blue_tick'].value_counts(), len(df['blue_tick']),
                  'Blue Tick Distribution', 'Blue Tick', 'Count', 3)
    # Email Verified plot
    variable_plot(df['email_verified'].value_counts(), len(df['email_verified']),
                  'Email Verified Distribution', 'Email Verified', 'Count', 4)
    # Gender plot
    variable_plot(df['gender'].value_counts(), len(df['gender']),
                  'Gender Distribution', 'Gender', 'Count', 5)
    # Email Type
    variable_plot(df['domain'].value_counts(), len(df['domain']),
                  'Email domain Distribution', 'Email Domain', 'Count', 6)
    # Number of follows
    multi_categories_plot(df['num_of_follows'].value_counts(),
                          'Number of Follows Distribution', 'Number of Follows', 'Count', 7)
    # Number of followers
    multi_categories_plot(df['num_of_followers'].value_counts(),
                          'Number of Followers Distribution', 'Number of Followers', 'Count', 8)
    # Number of previous messages
    multi_categories_plot(df['num_of_previous_messages'].value_counts(),
                          'Number of Previous Messages Distribution', 'Number of Previous Messages', 'Count', 9)
    # Year of user creation date
    variable_plot(df['year'].value_counts().sort_index(), len(df['year']),
                  'Year of User Creation Date Distribution', 'Year of User Creation Date', 'Count', 10)
    # Quarter of message date
    variable_plot(df['quarter'].value_counts().sort_index(), len(df['quarter']),
                  'Quarter of User Message Date Distribution', 'Quarter of User Message Date', 'Count', 11)
    # Text lengths
    multi_categories_plot(df['word_count'].value_counts().sort_index(),
                          'Distribution of Word Count in Text Column', 'Number of Words', 'Count', 12)
    # Sentiment labels
    variable_plot(df['sentiment'].value_counts().sort_index(), len(df['sentiment']),
                  'Sentiment Distribution', 'Sentiment', 'Count', 13)
    plt.show()


# Generate heat map to the correlations of the numeric variables
def heatmap_corr(df):
    columns_to_drop = ['textID', 'text', 'message_date', 'account_creation_date', 'previous_messages_dates',
                       'date_of_new_follower', 'date_of_new_follow', 'email', 'embedded_content', 'platform', 'domain',
                       'year']
    data_cop = df.copy()
    data_cop.drop(columns=columns_to_drop, inplace=True)
    plt.figure(14, figsize=(10, 6))
    sns.heatmap(data_cop.corr(), annot=True, cmap='coolwarm')
    plt.title('Heatmap of Numeric Variables')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    # plt.yticks(rotation=45)  # Rotate y-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()


''' Pre-processing Functions '''


# Preform per-processing to all the features in the data set
def pre_processing(df):
    df_cop = df.copy()
    df_cop['sentiment'] = df_cop['sentiment'].map({'positive': 1, 'negative': -1})
    df_cop['gender'] = df_cop['gender'].map({'F': 1, 'M': 0})
    df_cop['gender'] = df_cop['gender'].fillna(-1)
    df_cop['email_verified'] = df_cop['email_verified'].map({True: 1, False: 0}).combine_first(df_cop['email_verified'])
    df_cop['blue_tick'] = df_cop['blue_tick'].map({True: 1, False: 0}).combine_first(df_cop['blue_tick'])
    df_cop = convert_column_to_numerical(df_cop, 'domain')
    df_cop = convert_column_to_numerical(df_cop, 'platform')
    df_cop = convert_column_to_numerical(df_cop, 'year')
    df_cop = convert_column_to_numerical(df_cop, 'embedded_content')
    text_pre_processing(df_cop)
    return df_cop


# Text variable pre-processing functions
def text_pre_processing(df):
    df['text'] = df['text'].apply(remove_stopwords)
    df['text'] = df['text'].apply(remove_punctuation)
    df['text'] = df['text'].apply(lemmatize_text)
    df['text'] = df['text'].apply(lowercase_text)
    return df


# Remove stopwords from the text data
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


# Remove punctuation from the text data
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text


# Lematization
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word, wordnet.VERB) for word in words]
    return ' '.join(lemmatized_words)


# LowerCasing
def lowercase_text(text):
    lemmatizer = WordNetLemmatizer()
    # Tokenize the text into words
    words = text.split()
    # Lemmatize and lowercase each word
    lemmatized_words = [lemmatizer.lemmatize(word.lower(), wordnet.VERB) for word in words]
    return ' '.join(lemmatized_words)


# Convert columns of category to numbers for statistical computing
def convert_column_to_numerical(df, column_name):
    mapping = {value: i for i, value in enumerate(df[column_name].unique())}
    new_column_name = f'{column_name}_numerical'
    df[new_column_name] = df[column_name].map(mapping)
    return df


# Fill in missing value by column
def fill_missing_categorical_values(df, column_name):

    # Extract the non-null values from the specified column
    non_null_values = df[column_name].dropna()

    # Calculate the distribution of non-null values
    value_counts = non_null_values.value_counts(normalize=True)

    # Fill in the missing values based on the distribution of non-null values
    missing_values_count = df[column_name].isnull().sum()
    missing_values = np.random.choice(value_counts.index, size=missing_values_count, p=value_counts.values)

    # Fill in the missing values in the DataFrame
    df.loc[df[column_name].isnull(), column_name] = missing_values

    return df


# Handle missing values in the data set
def fill_values_df(df):
    df_cop = df.copy()
    df_cop = fill_missing_categorical_values(df_cop, 'email_verified')
    df_cop = fill_missing_categorical_values(df_cop, 'blue_tick')
    df_cop = fill_missing_categorical_values(df_cop, 'embedded_content_numerical')
    df_cop = fill_missing_categorical_values(df_cop, 'platform_numerical')
    df_cop = fill_missing_categorical_values(df_cop, 'domain_numerical')
    return df_cop


''' Feature Extraction Functions '''


def feature_extraction(df):
    df_cop = df.copy()
    df_cop = score_combined(df_cop, 9.3)
    df_cop = score_combined_engagement(df_cop, 18)
    df_cop = edge_platform(df_cop, 0.35)
    df_cop = add_sentiment_features(df_cop, 'text')
    return df_cop


# Kpi 1
def score_combined(df, score):
    df_cop = df.copy()
    df_cop['combined_score_connection'] = (0.4 * df_cop['num_of_previous_messages']) +\
                                          (0.3 * df_cop['num_of_follows']) + (0.3 * df_cop['num_of_followers'])
    # printing score combined
    num_samples_higher_than_9 = (df_cop['combined_score_connection'] > score).sum()
    print(f"\nSoical activities scores:\n{'=' * 60}")
    print(f"Number of samples with a combined_score_connection higher than {score}:", num_samples_higher_than_9)
    df_cop['social_connection'] = df_cop['combined_score_connection'].apply(lambda x: 1 if x >= score else 0)
    return df_cop


# Kpi 2
def score_combined_engagement(df, score):
    df_cop = df.copy()
    df_cop['combined_score_engagement'] = (0.5 * df_cop['word_count']) + (0.5 * df_cop['num_of_previous_messages'])
    # printing score combined
    num_samples_higher_than_9 = (df_cop['combined_score_engagement'] > score).sum()
    print(f"Number of samples with a combined_score_engagement higher than {score}:", num_samples_higher_than_9)
    df_cop['social_engagement'] = df_cop['combined_score_engagement'].apply(lambda x: 1 if x >= score else 0)
    return df_cop


# Search for edge platform and extract feature in case they are significant than corr score
def edge_platform(df, corr_score):
    df_cop = df.copy()
    # Group by 'platform' and calculate the mean sentiment for each platform
    mean_sentiment_by_platform = df_cop.groupby('platform_numerical')['sentiment'].mean()
    # Identify platforms that meet the condition
    platforms_meeting_condition = mean_sentiment_by_platform[mean_sentiment_by_platform.abs() > corr_score].index
    # Create separate binary columns for each platform meeting the condition
    for platform in platforms_meeting_condition:
        column_name = f'platform_{platform}_meets_condition'
        df_cop[column_name] = (df_cop['platform_numerical'] == platform).astype(int)
    # Print the mean sentiment for each platform
    print(f"\nMean Sentiment by Platform:\n{'=' * 60}")
    print(mean_sentiment_by_platform)
    return df_cop


def add_sentiment_features(df, text_column_name):
    df_cop = df.copy()
    # Check if the NLTK VADER lexicon is downloaded
    nltk.download('vader_lexicon')
    # Initialize the SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    # Extract sentiment features
    df_cop['compound_sentiment'] = df_cop[text_column_name].apply(lambda x: sia.polarity_scores(x)['compound'])
    df_cop['positive_sentiment'] = df_cop[text_column_name].apply(lambda x: sia.polarity_scores(x)['pos'])
    df_cop['negative_sentiment'] = df_cop[text_column_name].apply(lambda x: sia.polarity_scores(x)['neg'])
    df_cop['neutral_sentiment'] = df_cop[text_column_name].apply(lambda x: sia.polarity_scores(x)['neu'])
    # Add new columns based on sentiment comparisons
    df_cop['positive_text'] = (df_cop['positive_sentiment'] > df_cop['negative_sentiment']).astype(int)
    df_cop['negative_text'] = (df_cop['negative_sentiment'] > df_cop['positive_sentiment']).astype(int)
    top_positive_indices = df_cop['positive_sentiment'].nlargest(10).index
    print(f"\nTop 10 Positive Sentiment Entries:\n{'=' * 60}")
    for index in top_positive_indices:
        sentiment_label = 'Positive' if df_cop.loc[index, 'positive_text'] == 1 else 'Neutral/Negative'
        print(f"Index: {index}, Sentiment Label: {sentiment_label}, Text: {df_cop.loc[index, text_column_name]}")
    return df_cop


''' Features Representation Functions '''


def data_feature_represent(df):
    df_cop = df.copy()
    columns_to_drop = ['textID', 'text', 'message_date', 'account_creation_date', 'previous_messages_dates',
                       'date_of_new_follower', 'date_of_new_follow', 'email', 'embedded_content_numerical',
                       'platform_numerical', 'domain_numerical', 'year_numerical',
                       'combined_score_connection', 'combined_score_engagement', 'compound_sentiment',
                       'positive_sentiment', 'negative_sentiment', 'neutral_sentiment']
    df_cop.drop(columns=columns_to_drop, inplace=True)
    df_cop['gender'] = df_cop['gender'].map({1: 'female', 0: 'male', -1: 'other'}).infer_objects(copy=False)
    df_cop = encode_categorical_columns(df_cop, 'gender')
    df_cop = encode_categorical_columns(df_cop, 'embedded_content')
    df_cop = encode_categorical_columns(df_cop, 'platform')
    df_cop = encode_categorical_columns(df_cop, 'domain')
    df_cop = encode_categorical_columns(df_cop, 'year')
    df_cop = encode_categorical_columns(df_cop, 'quarter')
    df_cop = normalize_column_by_max(df_cop, 'num_of_follows')
    df_cop = normalize_column_by_max(df_cop, 'num_of_followers')
    df_cop = normalize_column_by_max(df_cop, 'num_of_previous_messages')
    df_cop = normalize_column_by_max(df_cop, 'word_count')
    return df_cop


def encode_categorical_columns(df, categorical_column):
    df_encoded = df.copy()
    # Use get_dummies to convert the categorical column into binary columns
    encoded_columns = pd.get_dummies(df[categorical_column], prefix=categorical_column, drop_first=False)
    encoded_columns = encoded_columns.astype(int)
    # Drop the original categorical column from the DataFrame
    df_encoded = df_encoded.drop(categorical_column, axis=1)
    # Concatenate the binary columns to the DataFrame
    df_encoded = pd.concat([df_encoded, encoded_columns], axis=1)
    return df_encoded


def normalize_column_by_max(df, column_name):
    df_normalized = df.copy()
    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        # Normalize the column by its maximum value
        max_value = df[column_name].max()
        df_normalized[column_name] = df[column_name] / max_value

    return df_normalized


''' Features Selection Functions '''


def remove_min_top_fvalues(df, target_column, top_n):
    df_cop = df.copy()
    # Drop rows with NaN values in any column
    df_cleaned = df_cop.dropna()

    # Extract features and target
    x = df_cleaned.drop(columns=[target_column])
    y = df_cleaned[target_column]

    # Perform F-test
    f_values, p_values = f_regression(x, y)

    # Initialize a dictionary to store feature names and F-values
    fvalues_dict = {feature: f_value for feature, f_value in zip(x.columns, f_values)}

    # Sort the dictionary by F-values in descending order
    sorted_fvalues = sorted(fvalues_dict.items(), key=lambda t: t[1])

    # Print header
    print(f"\n{target_column}: Top {top_n} Lowest F-Values\n{'=' * 60}")

    # Print the top N features with greatest F-values
    for i in range(min(top_n, len(sorted_fvalues))):
        feature, f_value = sorted_fvalues[i]
        print(f"{feature}: F-Value = {f_value:.4f}")
        df_cop.drop(columns=[feature], inplace=True)

    return df_cop


''' Denominational Reduction Functions '''


def pca(data, data1):
    x_selected = data[data1]
    m_pca = PCA(n_components=8)
    m_pca.fit(x_selected)
    explained_variance_ratio = m_pca.explained_variance_ratio_
    print(f"\nPCA variance results:\n{'=' * 60}")
    for i, ratio in enumerate(explained_variance_ratio):
        print("Explained variance ratio of PC{}: {:.2f}%".format(i + 1, ratio * 100))


''' Data edits '''
# Data load and edits
tr = pd.read_pickle('c:\Users\idodw\PycharmProjects\ML\XY_train.pkl')
trans_data = raw_data_trans(tr)

''' Pre-processing '''
proc_data = pre_processing(trans_data)

''' Data plots '''
# raw_data_plots(tr_data)
# heatmap_corr(proc_data)

filled_df = fill_values_df(proc_data)

''' Segmentation '''
# print(data[['email', 'domain']])

''' Feature Extraction '''
fe_df = feature_extraction(filled_df)
fe_df.to_excel(r'C:\Users\אופיר גוטליב\PycharmProjects\ml_proj\datasets\fe.xlsx', index=False)

''' Features Representation '''
fr_data = data_feature_represent(fe_df)
fr_data.to_excel(r'C:\Users\אופיר גוטליב\PycharmProjects\ml_proj\datasets\fr.xlsx', index=False)

''' Features Selection '''
fs_data = remove_min_top_fvalues(fr_data, 'sentiment', 10)
fs_data.to_excel(r'C:\Users\אופיר גוטליב\PycharmProjects\ml_proj\datasets\fs.xlsx', index=False)

''' Denominational Reduction '''
pca(fr_data, list(fs_data.columns))

# FE 2
# datat = tr
# duplicates = data.duplicated(subset=['gender'], keep=False)
# if duplicates.any():
#     print(data[duplicates])
