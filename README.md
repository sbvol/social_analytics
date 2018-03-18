# Demonstrate knowledge of APIs for Twitter and the data analysis that's possible.

### Loading Dependencies
***


```python
# Dependencies
import tweepy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```

### Initializing Variables
***


```python
# Target Search Term
target_terms = ("@BBCWorld", "@CBSNews", "@CNNbrk",
                "@FoxNews", "@nytimes")
len_targets = len(target_terms)
save_compound_list = []
save_sentiment = []
save_compound_list_with_date = []
counter = 0
y_axis = np.arange(0,100,1)
sentiment_array = []
```

### Twitter API Auth Steps
***


```python
# Twitter API Keys
consumer_key = 'ibYsFmAdHS8fhnupeA3opTRHN'
consumer_secret = 'cSfliluzlYkSJ8EPJEOvQA5kKG9BE6MG1ddF8kHAyNc5ZJ6601'
access_token = '943311356534472704-3tToKzZ2RMDtNOo4frlY6IEAg6iWGL1'
access_token_secret = '42jS6EeOwV5ZaWde0LwxpL4dPyozVt5rv3URu7ZlP9m17'
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Loop through all target users
for target in target_terms:

    # Variables for holding sentiments
    compound_list = []
    compound_list_with_date = []
    positive_list = []
    negative_list = []
    neutral_list = []


    # Using a loop to manage iteration thru each user
    if counter <= len_targets:

        # Use API to grab 100 tweets
        public_tweets = api.user_timeline(target, count=100)

        
        # Loop through all tweets
        for tweet in public_tweets:
            
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]

            
            compound_list.append(compound)
            compound_list_with_date.append(compound)
            compound_list_with_date.append(tweet['created_at'])
            compound_list_with_date.append(target)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)

    
    sentiment = {"User": target,
                 "Compound": np.mean(compound_list),
                 "Positive": np.mean(positive_list),
                 "Neutral": np.mean(negative_list),
                 "Negative": np.mean(neutral_list),
                 "Tweet Count": len(compound_list)}

    save_sentiment.append(sentiment)
    save_compound_list.append(compound_list)
    save_compound_list_with_date.append(compound_list_with_date)
    counter = counter + 1

```

### Getting the dataframes ready for plotting
***


```python
# Using the transpose function to make the rows columns
compound_df = pd.DataFrame(save_compound_list)
compound_df = compound_df.transpose()
compound_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.6486</td>
      <td>0.1779</td>
      <td>-0.4404</td>
      <td>-0.6705</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3412</td>
      <td>-0.6908</td>
      <td>0.4973</td>
      <td>0.5719</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>-0.6705</td>
      <td>0.2960</td>
      <td>0.6908</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.7783</td>
      <td>0.0000</td>
      <td>-0.5994</td>
      <td>0.4585</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>-0.1280</td>
      <td>-0.7003</td>
      <td>-0.0516</td>
      <td>-0.2960</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Setting the columns names
compound_df.columns = target_terms
compound_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>@BBCWorld</th>
      <th>@CBSNews</th>
      <th>@CNNbrk</th>
      <th>@FoxNews</th>
      <th>@nytimes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.6486</td>
      <td>0.1779</td>
      <td>-0.4404</td>
      <td>-0.6705</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3412</td>
      <td>-0.6908</td>
      <td>0.4973</td>
      <td>0.5719</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>-0.6705</td>
      <td>0.2960</td>
      <td>0.6908</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.7783</td>
      <td>0.0000</td>
      <td>-0.5994</td>
      <td>0.4585</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>-0.1280</td>
      <td>-0.7003</td>
      <td>-0.0516</td>
      <td>-0.2960</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Creating a 'index' so that plotting is easier
compound_df = compound_df.reset_index()
```


```python
# Plotting one of the users...but I have 5 total users
compound_df.plot(kind='scatter', x='index', y='@BBCWorld', subplots=False)
plt.show()
```


![png](output_13_0.png)


### Scatter Plotting 'Compound' Sentiment Analysis for ALL Datapoints
***


```python
# Using this plotting method given the number of variables/users
# Note that the plot legend is not on the graph
fig, ax = plt.subplots(sharex=True, figsize=(8, 6))
compound_df.plot(x='index', y='@BBCWorld', markersize=5, color='blue', linestyle='none', ax=ax, marker='o')
compound_df.plot(x='index', y='@CBSNews', markersize=5, color='red', linestyle='none', ax=ax, marker='o')
compound_df.plot(x='index', y='@CNNbrk', markersize=5, color='green', linestyle='none', ax=ax, marker='o')
compound_df.plot(x='index', y='@FoxNews', markersize=5, color='black', linestyle='none', ax=ax, marker='o')
compound_df.plot(x='index', y='@nytimes', markersize=5, color='orange', linestyle='none', ax=ax, marker='o')
plt.ylabel("Tweet Polarity", size=10)
plt.grid(True, color='gray', linestyle='-', linewidth=.5)
plt.xlabel("Tweets Ago", size=10)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Sentiment Analysis of Media Tweets', size=17)
plt.show()
```


![png](output_15_0.png)


### Extra Credit - Plotting TimeSeries Data
***


```python
# This was a neat exercise of learning how to convert a list into a 3 column dataframe
# Using the reshape function is required before the column names can be assigned
compound_wdate_df0 = pd.DataFrame(np.array(save_compound_list_with_date[0]).reshape(100,3), columns = ['Compound', 'Date', 'User'])
compound_wdate_df1 = pd.DataFrame(np.array(save_compound_list_with_date[1]).reshape(100,3), columns = ['Compound', 'Date', 'User'])
compound_wdate_df2 = pd.DataFrame(np.array(save_compound_list_with_date[2]).reshape(100,3), columns = ['Compound', 'Date', 'User'])
compound_wdate_df3 = pd.DataFrame(np.array(save_compound_list_with_date[3]).reshape(100,3), columns = ['Compound', 'Date', 'User'])
compound_wdate_df4 = pd.DataFrame(np.array(save_compound_list_with_date[4]).reshape(100,3), columns = ['Compound', 'Date', 'User'])
```


```python
compound_sub1 = compound_wdate_df0.append(compound_wdate_df1)
```


```python
compound_sub2 = compound_wdate_df2.append(compound_wdate_df3)
```


```python
compound_sub2 = compound_sub2.append(compound_wdate_df4)
```


```python
compound_final = compound_sub1.append(compound_sub2)
```


```python
# All of the steps above a needed to build a combined dataframe
# Notice also that the date is a column
compound_final.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.6486</td>
      <td>Mon Jan 15 01:56:42 +0000 2018</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3412</td>
      <td>Mon Jan 15 01:52:15 +0000 2018</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>Mon Jan 15 01:46:24 +0000 2018</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.7783</td>
      <td>Mon Jan 15 01:42:25 +0000 2018</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>Mon Jan 15 01:35:38 +0000 2018</td>
      <td>@BBCWorld</td>
    </tr>
  </tbody>
</table>
</div>




```python
compound_final['Date'] = pd.to_datetime(compound_final['Date'])
```


```python
# This step is needed to get the date into the index
# This is required if you want to easily plot the data
compound_final.set_index(['Date'], inplace=True)
compound_final.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>User</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-15 01:56:42</th>
      <td>-0.6486</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>2018-01-15 01:52:15</th>
      <td>0.3412</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>2018-01-15 01:46:24</th>
      <td>0.0</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>2018-01-15 01:42:25</th>
      <td>-0.7783</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>2018-01-15 01:35:38</th>
      <td>0.0</td>
      <td>@BBCWorld</td>
    </tr>
  </tbody>
</table>
</div>




```python
# This graph is the right data but the x and y axis are not on the right scale. Notice the date span on the x axis
# Notice the y axis labels
plt.scatter(compound_final.index, compound_final.Compound)
plt.show()
```


![png](output_25_0.png)


### Plotting the Average Compound Value for Each User
***


```python
# Creating a sentiment dataframe
# Writing sentiment data to a csv file
sentiment_df = pd.DataFrame(save_sentiment)
sentiment_df.to_csv('sentiment.csv', sep=',', header=True, index=True, index_label=None)
sentiment_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweet Count</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.142123</td>
      <td>0.81369</td>
      <td>0.13182</td>
      <td>0.05448</td>
      <td>100</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.156360</td>
      <td>0.84051</td>
      <td>0.11656</td>
      <td>0.04289</td>
      <td>100</td>
      <td>@CBSNews</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.035990</td>
      <td>0.81470</td>
      <td>0.08281</td>
      <td>0.10250</td>
      <td>100</td>
      <td>@CNNbrk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.024568</td>
      <td>0.83032</td>
      <td>0.09784</td>
      <td>0.07186</td>
      <td>100</td>
      <td>@FoxNews</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.041352</td>
      <td>0.86409</td>
      <td>0.07588</td>
      <td>0.05999</td>
      <td>100</td>
      <td>@nytimes</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Note that the colors for each bar are the same
colors = ['#624ea7', 'g', 'yellow', 'k']
sentiment_df.plot.bar(x='User', y='Compound', subplots=False, color=colors)
plt.grid(True, color='gray', linestyle='-', linewidth=.5)
plt.axhline(0, color='k')
plt.ylabel("Tweet Polarity", size=10)
plt.xlabel("User", size=10)
plt.title('Sentiment Analysis of Media Tweets', size=17)
plt.show()
```


![png](output_28_0.png)



```python
# Using different code to change the color of each bar
# Also notice the highlighted '0' line
n=5
tick_label = sentiment_df['User']
data = sentiment_df['Compound']
fig, ax = plt.subplots(figsize=(7, 5))
bar_locations = np.arange(n)
colors = ['#624ea7', 'g', 'yellow', 'k', 'maroon']
plt.grid(True, color='gray', linestyle='-', linewidth=.5)
plt.axhline(0, color='k')
ax = plt.bar(bar_locations, data, color=colors, tick_label=tick_label)
plt.ylabel("Tweet Polarity", size=10)
plt.xlabel("Users", size=10)
plt.title('Sentiment Analysis of Media Tweets', size=17)
plt.show()
```


![png](output_29_0.png)


### Extra Credit - Adding a data table to the bottom of my graph
***


```python
# With more time I would have liked to have experimented with this method
data = sentiment_df['Compound']
columns = target_terms
rows = 'Compound'

values = np.arange(-.175, .150, .025)

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = 1

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % x for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowColours=colors,
                      colLabels=columns)

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

#plt.ylabel("Loss in ${0}'s".format(value_increment))
#plt.yticks(values ['%d' % val for val in values])
#plt.xticks([])
plt.title('Title')
plt.show()
```


![png](output_31_0.png)



```python
# Adding data labels to the graph using the zip feature
x = sentiment_df['User']
y = sentiment_df['Compound']
plt.bar(x, y)
for a,b in zip(x, y):
    plt.text(a, b, str(b))
colors = ['#624ea7', 'g', 'yellow', 'k', 'maroon']
plt.grid(True, color='gray', linestyle='-', linewidth=.5)
plt.axhline(0, color='k')
plt.ylabel("Tweet Polarity", size=10)
plt.xlabel("Users", size=10)
plt.title('Sentiment Analysis of Media Tweets', size=17)
plt.show()
```


![png](output_32_0.png)


### Obversations
***

On the surface this appeared to be a relatively straight forward assignment.  There were challenges with graphs (data labels and changing the color of the each bar for example).  There were challenges with the list and dataframes including working with DateTime, index, adding rows to dataframes, etc.

Interesting that only one news agency had a positive sentiment. It would be interesting to do more analysis on the
CBS tweets to determine why this user is far less than 2 of the others...why it's the worse score.

With more time I would have loved to have down more analysis on the positive sentiment tweets to see which words or which subjects generated positive sentiments.
