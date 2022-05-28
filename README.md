# Deep Learning for Detecting Mental Health Disorders using Social Media Generated Content

## What is this Project?
Master of Science in Artificial Intelligence, Queen Mary, University of London, 2020-2021 Dissertation Code.

##  Project Aims

The advent of social media has enabled new forms of
socialisation which have become increasingly essential to the
daily lives of millions of people. Many people, using various
social media platforms have been experienced an
unprecedented level of connectivity. Social Networking sites
have gained incredible prominence with Facebook registering
2,740,000,000 million users as of January 2021. (statista,
2021)
Social media is fundamentally an interactive technology
which allows users to create and share information. This user-
generated content primarily consists of text-based posts and/or
comments, possibly accompanied with digital photos or
videos, and the corresponding metadata. Such broad use has
enabled increased communication, allowing people to stay up
to date with family and friends, join and promote social
causes, and consume content of textual and graphic nature.
Social media has also facilitated damaging forms of
interaction such as cyber-texting, sexting, and online stalking.
Notably, excessive use of these online platforms has been
shown to fuel feelings of Anxiety, Depression, isolation, and
FOMO (fear of missing out). (Obar & Wildman, 2015).
Mental health awareness has increased significantly in
recent years, owing its origins to the mental hygiene
movement, initiated in 1908, which was driven by a desire to
improve the treatment and quality of life of people with mental
disorders. (Bertolote, 2008) With approximately 1 in 4 people
in the UK suffering from a mental condition every year, and 1
in 6 suffering from a common disorder such as Anxiety or
Depression, such issues have become widespread and
prevalent.
It has been noted that people are commonly readily eager
to express their views anonymously online rather than in
person. (Al-Saggar & Nielsen, 2014) Using various
applications, anonymous users are likely to discuss their
mental health problems on an online platform. (Hanwen Shen
& Rudzicz, 2017) This is a positive trend due to patterns of
hidden behaviours exhibited by people when fearing
stigmatisation, i.e. people have been more likely to under
report mental health issues compared to other health
conditions due to the associated stigma. (Bharadwaj &
Suziedelyte, 2017) Therefore, the anonymisation of identity,
allowed by online interaction, has allowed people to discuss
their mental issues without risking social stigma. By
discussing their issues openly and without fear of
stigmatisation, people have been less likely to under-report
and therefore, have been giving more accurate accounts of
potential symptoms. People may also engage in discussions
but be unaware or not willing to consider that they are
potentially suffering from a mental health condition.
It is unrealistic to expect mental health professionals to
inspect and review large and ever-increasing amounts of data.
An automated unit for the purposes of mental health text
classification could alert online users if the text they are
posting would be indicative of a mental health disorder and
would encourage them to seek professional help.
This project focused on using the social media generated
textual data to train a Deep Learning based model with the aim
of detecting whether the person posting on social media is
likely to be suffering from a mental health disorder based on
the content they are posting onto social media. Such a task
would fall into the field of Natural Language Processing
(NLP). Core NLP techniques were traditionally dominated by
machine learning methods using linear methods such as
support vector machines or linear regression, trained over very
high dimensional yet sparse feature vectors. The field has
since found increased success in recent years by making use
of non-linear neural network models over dense inputs, a
technique known as Neural Networks. (Goldberg, 2015)
Such a technique, specifically when extended to Deep
Learning, allows computational models that are composed of
multiple processing layers to develop representations of data
with various levels of abstraction. Deep Learning models have
dramatically improved the state-of-the-art in various domains
of application. The strength of these models is their capability
to discover intricate structure and nuanced patterns in large
datasets. This is accomplished by using the backpropagation
algorithm to indicate how a machine should change its
learnable weights which are used to compute the
representation in each layer from the representation in theprevious layer. (LeCun, et al., 2015) The application of Deep
Learning to Natural Language Processing tasks yields several
advantages: superior performance at pattern recognition tasks,
and the capability of end-to-end training (little or no domain
knowledge is needed prior to the system construction).
Deep Learning however is a data hungry process and is
hence not suitable for small quantities of data. Its resultant
models are typically black box, making them difficult to
understand due to the continuing lack of theoretical
foundation. Furthermore, the cost of training Deep Learning
models is computationally expensive. (Li, 2018)
This work relies on the notion that the text posted by a user
when suffering from a mental health condition would contain
different, detectable features when compared to the text
generated by the same user when not suffering from a mental
health condition.

## Project Structure

### 1.Development
This section includes the files necessary to train the mental health classifiers. 

### 2.Deployment
This section includes the files used to deployment a Gradio based web demo and a terminal based demo using flask.

### 3.Documentation
This section includes deliverables including the Dissertation paper discussing the project, a reflective essay focusing more on ethics, and potential social impacts of such a project, and finally a power point presentation briefly discussing an overview of the project.