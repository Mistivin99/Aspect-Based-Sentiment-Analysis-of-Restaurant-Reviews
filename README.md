# Aspect-Based-Sentiment-Analysis-of-Restaurant-Reviews

**General Objective**
The objective of our project is to leverage advanced language models and sentiment analysis techniques to extract valuable insights from restaurant reviews. Specifically, we aim to analyze various aspects of the dining experience, such as food quality, interior ambiance, and service, by processing textual feedback provided by customers. By understanding and quantifying customer sentiments expressed in these reviews, our goal is to provide actionable information to businesses in the hospitality industry, enabling them to improve their services, enhance customer satisfaction, and ultimately thrive in a competitive market landscape.

**Libraries and Models used**
**i)	For extraction of insights:**

**•	Langchain Library:** The Langchain library is a Python library designed to facilitate the interaction with language models and streamline text generation tasks. It provides a set of functionalities and tools to interact with various language models, enabling users to perform tasks such as text generation, sentiment analysis, summarization, and more. Specifically, it uses the CTransformers class from the langchain library for interacting with language models. This class offers methods and utilities for loading pre-trained models, generating text based on prompts, fine-tuning models, and performing various natural language processing tasks.

**•	Pandas Library: **Pandas is used for data manipulation tasks, including loading the dataset containing restaurant reviews, preprocessing the data, and extracting relevant information from the reviews.

**•	Mistral-7B-Instruct language model :** Specifically, we use the ‘Mistral-7B-Instruct-v0.1-GGUF’ model from TheBloke, which is accessed through the CTransformers library. This language model is specifically trained and optimized for instruction-based text generation tasks, making it suitable for generating responses to prompts and extracting insights from the context provided in the restaurant reviews with higher accuracy. The model has the capability to understand and generate responses based on the context provided in the prompts. This enables it to generate relevant and accurate insights from the textual data, catering to the specific requirements of the project.

**ii)	 For sentiment analysis: **

**•	Transformers library: **  The Transformers library, developed by Hugging Face, offers a comprehensive suite of state-of-the-art pre-trained models for natural language processing tasks. With a unified interface and a vast model hub, it enables seamless experimentation and deployment of models for various NLP tasks, including text classification, summarization, translation, and more. Its support for fine-tuning and transfer learning facilitates adaptation of pre-trained models to specific tasks or domains, while its integration with PyTorch and TensorFlow ensures compatibility with popular deep learning frameworks.
Specifically, it uses the pipeline function from the transformers library to create a sentiment analysis pipeline. This pipeline utilizes pre-trained models for sentiment analysis tasks.

**•	BERTweet Model: **  The sentiment analysis pipeline likely utilizes the 'finiteautomata/bertweet-base-sentiment-analysis' model for sentiment analysis. This model is pre-trained on Twitter data and is capable of analyzing sentiment in text data. BERTweet is tailored to handle the unique characteristics of tweets, including their informal language, abbreviations, emoticons, hashtags, and URLs. By training on a large corpus of Twitter text, BERTweet learns to effectively capture the nuances and context of tweets, enabling it to perform tasks such as sentiment analysis, classification, and entity recognition with high accuracy on Twitter data. This makes BERTweet particularly useful for analyzing sentiment and extracting insights from tweets in various applications, including social media monitoring, opinion mining, and trend analysis.

**About the Dataset**
The dataset, sourced from **blinoff/restaurants_reviews**, hugging face, comprises of user-generated reviews about restaurants. It encompasses a total of 47,139 reviews, each offering insights into customers' dining experiences.
Each review within the dataset is categorized based on its overall sentiment, as well as sentiments specific to three key aspects: food, interior, and service. These categorizations provide a nuanced understanding of customers' perceptions and preferences across different facets of their dining encounters.


