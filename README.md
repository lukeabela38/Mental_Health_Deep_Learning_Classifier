# Deep Learning for Detecting Mental Health Disorders using Social Media Generated Content

## What is this Project?
Master of Science in Artificial Intelligence, Queen Mary, University of London, 2020-2021 Dissertation Code.

##  Project Aims

The drastic increase in use of Social Media over the last few years has occurred
concurrently with an increase in attention given to issues pertaining to mental health
disorders. This dissertation aims to use social-media generated content to develop a
method based on a deep learning framework to detect whether someone is likely to
have a mental health condition based on the text they post.
Such data may be obtained from various sites such as Twitter or Reddit. Twitter, though
popular, places an arbitrary character limit on each person’s individual tweets. Reddit, a
forum based discussion site, features lengthier posts. Reddit is composed of a series of
‘subReddits’, a dedicated page for the discussion of a given topic between users. There
exist subReddits for mental health disorders, such as r/Anxiety and r/Depression.
This dissertation will focus on using Reddit-sourced textual data to train a set of binary-
classifiers to detect whether someone is suffering from a given disorder such as anxiety
or depression. Each disorder tested for would have its own individual classifier and the
detection of multiple disorders in a single user could be accomplished by combining the
outputs of these binary classifiers.

## Project Structure

### 1.Development
This section includes the files necessary to train the mental health classifiers. 

### 2.Deployment
This section includes the files used to deployment a Gradio based web demo and a terminal based demo using flask.