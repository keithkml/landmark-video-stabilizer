"""Real encoded-video checks and regression of the shipped calibration."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from landmark_stabilizer.cli import main
from landmark_stabilizer.wobble import cli, wobble


def test_bundled_calibration_preserves_code_and_reference_predictions():
    model = cli.read(cli.DEFAULT_MODEL)
    provenance = cli.read(cli.DEFAULT_MODEL.with_name('provenance.json'))
    assert cli.digest(cli.DEFAULT_MODEL) == provenance['model_sha256']
    assert cli.signature() == model['extractor_sha256'] == provenance['extractor_sha256']
    assert cli.predictor_signature() == model['predictor_sha256'] == provenance['predictor_sha256']
    fixture = cli.read(Path(__file__).parent / 'fixtures/wobble-native-v3.json')
    expected = {r['id']: r for r in fixture['expected']['resubstitution']['predictions']}
    for example in fixture['examples']:
        decision = wobble.predict_features(example['features'], model)
        assert decision['prediction'] == expected[example['id']]['prediction']
        assert decision['score'] == expected[example['id']]['score']
    result = wobble.evaluate_examples(fixture['examples'], fixture['pairs'], model['feature_names'])
    assert result['model']['trees'] == model['trees']
    for key, reference in fixture['expected'].items():
        assert result[key] == reference
    assert result['group_holdout']['correct'] == 6
    assert result['group_holdout']['total'] == 9
    assert result['acceptance_gate']['automatic_approval_permitted'] is False


@pytest.fixture(params=['textured', 'flat'])
def encoded_video(request, tmp_path):
    path = tmp_path / (request.param + '.avi')
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 24, (320, 240))
    assert writer.isOpened(), 'MJPEG encoder is needed for end-to-end tests'
    rng = np.random.default_rng(543)
    frame = (rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
             if request.param == 'textured' else np.full((240, 320, 3), 127, np.uint8))
    try:
        for _ in range(60):
            writer.write(frame)
    finally:
        writer.release()
    return path, request.param


def test_main_cli_decodes_real_video_with_default_model(encoded_video, tmp_path, capsys):
    video, kind = encoded_video
    original = cli.digest(video)
    output = tmp_path / 'result.json'
    main(['wobble', str(video), '--output', str(output)])
    stdout = json.loads(capsys.readouterr().out)
    result = cli.read(output)
    assert stdout['sha256'] == result['sha256'] == original == cli.digest(video)
    assert result['automatic_approval_permitted'] is False
    assert result['analysis']['sampling']['sampled_frames'] == 60
    assert result['analysis']['sampling']['decoded_frames'] == 60
    if kind == 'flat':
        assert result['prediction'] == 'needs_review'
        assert result['score'] is None
    else:
        assert result['analysis']['quality']['usable']
        assert result['prediction'] in {'too_wobbly', 'not_too_wobbly'}
        assert 0 <= result['score'] <= 1


def test_standalone_cli_defaults_to_bundled_model(encoded_video, capsys):
    video, _ = encoded_video
    cli.main(['detect', str(video)])
    result = json.loads(capsys.readouterr().out)
    assert result['model_sha256'] == cli.digest(cli.DEFAULT_MODEL)
    assert result['automatic_approval_permitted'] is False


def test_detect_cannot_overwrite_media_or_results(tmp_path):
    source = tmp_path / 'camera.mp4'
    source.write_bytes(b'original camera bytes')
    prior = tmp_path / 'result.json'
    prior.write_text('previous report')
    link = tmp_path / 'alias.json'
    link.symlink_to(source)
    for output in [source, prior, link]:
        with pytest.raises(FileExistsError):
            cli.detect(SimpleNamespace(video=source, output=output, model=cli.DEFAULT_MODEL))
    assert source.read_bytes() == b'original camera bytes'
    assert prior.read_text() == 'previous report'


@pytest.mark.parametrize('field', ['extractor_sha256', 'predictor_sha256'])
def test_changed_implementation_binding_is_rejected_before_opening_video(tmp_path, field):
    model = deepcopy(cli.read(cli.DEFAULT_MODEL))
    model[field] = '0' * 64
    path = tmp_path / 'changed-model.json'
    cli.write(path, model)
    with pytest.raises(ValueError, match='code differs'):
        cli.detect(SimpleNamespace(model=path, video=tmp_path / 'absent.mp4', output=None))


def test_bad_input_fails_without_creating_a_report(tmp_path, capsys):
    source = tmp_path / 'invalid.mp4'
    source.write_text('not a video')
    output = tmp_path / 'report.json'
    with pytest.raises(SystemExit) as exc:
        main(['wobble', str(source), '--output', str(output)])
    assert exc.value.code == 2
    assert not output.exists()
    assert 'error:' in capsys.readouterr().err
