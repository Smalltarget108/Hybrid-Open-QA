import jsonlines

with jsonlines.open('/home/ec2-user/efs/ott-qa/hystruct_es_retrieval-ottqa-provided-supp-20210130T141346-YJY/dev.traced.es_retrieved.jsonl') as f:
    data = [line for line in f.iter()]

for line in data:
    for key in ['supp_tables', 'supp_passages']:
        for x in line[key]:
            x['judge'] = 1
            x['es_score'] = 100

    line['tables'] = line['supp_tables'] + line['tables']
    line['passages'] = line['supp_passages'] + line['passages']

with jsonlines.open('/home/ec2-user/efs/ott-qa/hystruct_es_retrieval-ottqa-provided-supp-20210130T141346-YJY/ott-qa.dev.es_retrieved.processed.jsonl', 'w') as writer:
    writer.write_all(data)
