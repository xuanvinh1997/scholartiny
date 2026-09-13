"""Original synthetic fixtures and inherited RAG splits, no network."""
import json
from scripts.generate_academic import generate_records
from scripts.build_rag_sft import build
from agent.tools import compute_math
from dataset.protocol import canonical_json


def test_synthetic_covers_all_families_and_observations_are_computed():
    rows = list(generate_records(28, seed=123))
    assert {row['task_family'] for row in rows} == {str(i) for i in range(7)}
    for row in rows:
        messages = row['messages']
        for i, message in enumerate(messages):
            if message['role'] == 'assistant' and message.get('tool_calls'):
                call = message['tool_calls'][0]
                observation = messages[i + 1]
                assert observation['role'] == 'tool'
                if call['name'].startswith('math.'):
                    assert json.loads(observation['content'])['result'] == compute_math(call['name'], call['arguments'])


def test_synthetic_has_real_second_tool_turn():
    row = next(r for r in generate_records(14, seed=123) if r['task_family'] == '6')
    assert [m['role'] for m in row['messages']] == ['user', 'assistant', 'tool', 'assistant', 'tool', 'assistant']
    first = json.loads(row['messages'][2]['content'])['result']['exact']
    second_expression = row['messages'][3]['tool_calls'][0]['arguments']['expression']
    assert second_expression.startswith(first + '*')


def test_rag_warmup_inherits_license_split_and_document_group(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    originals = {}
    for index, split in enumerate(('train', 'val', 'test')):
        row = {'id': f'{index + 1:064x}', 'group_id': f'group-{split}', 'split': split,
               'kind': 'pretrain', 'text': (f'Original {split} fixture document. ' * 8),
               'provenance': {'title': f'Document {split}', 'card_licenses': ['cc-by-sa-3.0'],
                              'url': 'https://example.invalid/fixture'}}
        originals[split] = row
        (source / (split + '.jsonl')).write_text(canonical_json(row) + '\n', encoding='utf-8')
    out = tmp_path / 'output'
    assert build(source, out, limit=10) == {'train': 2, 'val': 2, 'test': 2}
    for split in originals:
        pairs = [json.loads(line) for line in (out / (split + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        for row in pairs:
            assert row['split'] == split
            assert row['group_id'] == originals[split]['group_id']
            assert row['parent_document_id'] == originals[split]['id']
            assert row['provenance'] == originals[split]['provenance']
            evidence = json.loads(row['messages'][2]['content'])['result']
            if row['missing_evidence']:
                assert evidence == []
                assert 'Chưa đủ bằng chứng' in row['messages'][-1]['content']
            else:
                assert evidence[0]['text'].startswith(f'Original {split}')
                assert '[' + evidence[0]['chunk_id'] + ']' in row['messages'][-1]['content']
