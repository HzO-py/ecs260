# Diversity in LLM-Simulated Survey Responses: A Research based on Stack Overflow Survey Questions

## Steps to run this code

1. Download dataset: https://cdn.sanity.io/files/jo7n4k8s/production/262f04c41d99fea692e0125c342e446782233fe4.zip/stack-overflow-developer-survey-2024.zip

2. Clone the code repository and place survey_results_schema.csv and survey_results_public.csv from the dataset into the project directory.

3. Link to the extracted indexes(compressed), please uncompress and place this indexes folder inside the dataset folder: https://drive.google.com/file/d/1WfLOnpHwQibGF--W0VpoLhJO-aanIEMl/view?usp=drive_link. It is based on the processed  RAG vector dataset mentioned below to save time for RAG.

4. Update schema_path,public_path, index_path,and key with your openai api key in main.py.

5. Create output folder and update path in main.py.

5. Based on your requirement, run the zero-shot or RAG as prompt strategy choosing llama, deepseek or chatgpt as using LLM. The using of chatgpt and deepseek needs to open the corresponding file and comment out part of the code according to the comment, and uncomment part of the code. Besides, if you want to use deepseek or llama,you should open the corresponding file and input your LLM key.

6. Based on your requirement, compare different distribution

7. Run main.py.

## Related resources

For the RAG vector dataset, here is the link for the original dataset: https://archive.org/details/stackexchange_20240930

Here you can download the processed RAG vector dataset: https://drive.google.com/drive/folders/1GYi4TZYikjyATNGGJUz8vfDM7SuCAP9U?usp=drive_link

Another link for the processed RAG vector dataset(compressed): https://drive.google.com/file/d/1MO6n8UKOHKBljRAqEsXt8Wq2nxw6LLsC/view?usp=drive_link
