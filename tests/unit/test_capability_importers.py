"""Capability dataset importers: synthetic fixtures per documented release format.

Each test builds the smallest valid form of the real release format, imports it,
and validates the resulting manifest. No real dataset files are bundled.
"""

import json
from pathlib import Path

import pytest
from aurora_visual.training.data import load_manifest
from aurora_visual.training.importers import (
    CAPABILITY_SOURCES,
    CAPTION_SOURCES,
    import_capability,
    import_caption_dataset,
    load_capability_manifest,
)
from PIL import Image


@pytest.fixture
def image_root(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    return root


def add_image(root: Path, relative: str, color="#3366cc", size=(40, 30)):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return relative


LICENSE = "Research access approved; original media rights retained"


def test_snli_ve_import_emits_weak_three_state_labels(tmp_path, image_root):
    add_image(image_root, "flickr30k-images/123.jpg")
    source = tmp_path / "snlive.jsonl"
    entries = [
        {"caption": "A dog runs on grass", "image": "flickr30k-images/123.jpg", "label": "entailment"},
        {"caption": "A cat sleeps on grass", "image": "flickr30k-images/123.jpg", "label": "contradiction"},
        {"caption": "The photo was taken in 1990", "image": "flickr30k-images/123.jpg", "label": "neutral"},
    ]
    source.write_text("\n".join(json.dumps(entry) for entry in entries))
    output = tmp_path / "snli-ve.jsonl"
    result = import_caption_dataset("snli-ve", source, output, image_root, "train", LICENSE)
    assert result["count"] == 3
    records = load_manifest(output)
    labels = {a.label for record in records for a in record.annotations}
    assert labels == {"Supported", "Contradicted", "Unobservable"}
    assert all(a.provenance == "weak" for record in records for a in record.annotations)
    assert "entailment->Supported" in records[0].extensions["label_mapping"]
    with pytest.raises(ValueError, match="Unknown SNLI-VE label"):
        entries.append({"caption": "x", "image": "flickr30k-images/123.jpg", "label": "maybe"})
        source.write_text("\n".join(json.dumps(entry) for entry in entries))
        import_caption_dataset("snli-ve", source, output, image_root, "train", LICENSE)


def test_countbench_import_carries_count_metadata(tmp_path, image_root):
    add_image(image_root, "cb/0.jpg")
    source = tmp_path / "countbench.json"
    source.write_text(
        json.dumps({"entries": [{"caption": "Two dogs sit", "image_path": "cb/0.jpg", "number": 2}]})
    )
    result = import_caption_dataset(
        "countbench", source, tmp_path / "out.jsonl", image_root, "train", LICENSE
    )
    assert result["count"] == 1 and result["weak_atom_labels"] == 0
    records = load_manifest(tmp_path / "out.jsonl")
    assert records[0].extensions["count"] == {"number": 2, "source": "countbench-curated"}


def test_refcoco_import_normalizes_gold_regions(tmp_path, image_root):
    add_image(image_root, "ref/1.jpg", size=(100, 50))
    source = tmp_path / "refcoco.json"
    source.write_text(
        json.dumps(
            [
                {
                    "sentence": "the man in the red shirt",
                    "image": "ref/1.jpg",
                    "bbox": [10, 5, 50, 20],
                    "width": 100,
                    "height": 50,
                }
            ]
        )
    )
    import_caption_dataset("refcoco", source, tmp_path / "out.jsonl", image_root, "train", LICENSE)
    records = load_manifest(tmp_path / "out.jsonl")
    grounding = records[0].extensions["grounding"]
    assert grounding["bbox"] == [0.1, 0.1, 0.6, 0.5]
    assert grounding["provenance"] == "refcoco-gold-region"


def test_flickr30k_entities_import_with_phrase_boxes(tmp_path, image_root):
    add_image(image_root, "36979.jpg")
    source = tmp_path / "sentences.json"
    source.write_text(
        json.dumps(
            {
                "36979": [
                    {
                        "sentence": "A man wears a red shirt",
                        "phrases": [{"phrase": "red shirt", "phrase_id": "36979_0"}],
                    }
                ]
            }
        )
    )
    boxes = tmp_path / "Annotations"
    boxes.mkdir()
    (boxes / "36979.xml").write_text(
        "<annotation><object><name>36979_0</name>"
        "<bndbox><xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>25</ymax></bndbox>"
        "</object></annotation>"
    )
    import_caption_dataset(
        "flickr30k-entities", source, tmp_path / "out.jsonl", image_root, "train", LICENSE, boxes_dir=boxes
    )
    records = load_manifest(tmp_path / "out.jsonl")
    phrase = records[0].extensions["grounding"]["phrases"][0]
    assert phrase["phrase"] == "red shirt" and phrase["bbox"] == [10, 5, 30, 25]


def test_visual_genome_import_aggregates_regions_and_relations(tmp_path, image_root):
    add_image(image_root, "1.jpg", size=(100, 100))
    regions = tmp_path / "regions.json"
    regions.write_text(
        json.dumps(
            [
                {
                    "phrase": "a red ball",
                    "region": {"x": 10, "y": 10, "width": 20, "height": 20},
                    "image": {
                        "image_id": 1,
                        "url": "https://example.test/1.jpg",
                        "width": 100,
                        "height": 100,
                    },
                },
                {
                    "phrase": "on the table",
                    "region": {"x": 0, "y": 50, "width": 50, "height": 50},
                    "image": {
                        "image_id": 1,
                        "url": "https://example.test/1.jpg",
                        "width": 100,
                        "height": 100,
                    },
                },
            ]
        )
    )
    relations = tmp_path / "relationships.json"
    relations.write_text(
        json.dumps(
            [
                {
                    "predicate": "on",
                    "subject": {"phrase": "ball"},
                    "object": {"phrase": "table"},
                    "image": {"image_id": 1},
                }
            ]
        )
    )
    import_caption_dataset(
        "visual-genome",
        regions,
        tmp_path / "out.jsonl",
        image_root,
        "train",
        LICENSE,
        relations_path=relations,
    )
    records = load_manifest(tmp_path / "out.jsonl")
    assert len(records) == 1  # one Example per image, regions aggregated
    assert len(records[0].extensions["regions"]) == 2
    assert records[0].extensions["relations"][0]["predicate"] == "on"
    assert records[0].extensions["regions"][0]["bbox"] == [0.1, 0.1, 0.3, 0.3]


def test_sa1b_capability_import_bounded_masks(tmp_path, image_root):
    add_image(image_root, "sa/1.jpg", size=(100, 100))
    source = tmp_path / "sa1b.json"
    source.write_text(
        json.dumps(
            {
                "image_id": 1,
                "image": {"file_name": "sa/1.jpg", "height": 100, "width": 100},
                "annotations": [
                    {
                        "id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "segmentation": [[10, 10, 30, 10, 30, 30]],
                    },
                    {"id": 2, "bbox": [50, 50, 10, 10], "area": 100, "segmentation": []},
                ],
            }
        )
    )
    result = import_capability("sa1b", source, tmp_path / "out.jsonl", image_root, "train", LICENSE)
    assert result["manifest_kind"] == "capability"
    records = load_capability_manifest(tmp_path / "out.jsonl")
    capability = records[0].extensions["capability"]
    assert capability["kind"] == "segmentation" and capability["mask_count"] == 2
    assert capability["masks"][0]["bbox"] == [0.1, 0.1, 0.3, 0.3]


def test_fsc147_capability_import_with_classes(tmp_path, image_root):
    add_image(image_root, "images/1.JPG")
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps({"images/1.JPG": {"number": 27, "box": [10, 10, 30, 30]}}))
    classes = tmp_path / "ImageClasses.tsv"
    classes.write_text("images/1.JPG\ttomato\n")
    import_capability(
        "fsc-147", source, tmp_path / "out.jsonl", image_root, "train", LICENSE, classes_path=classes
    )
    records = load_capability_manifest(tmp_path / "out.jsonl")
    capability = records[0].extensions["capability"]
    assert capability == {
        "kind": "counting",
        "number": 27,
        "class": "tomato",
        "exemplar_box": [10, 10, 30, 30],
        "caption_template": "27 tomato",
    }


def test_im2gps3k_capability_import_by_line_index(tmp_path, image_root):
    add_image(image_root, "0.jpg")
    add_image(image_root, "1.jpg")
    source = tmp_path / "im2gps3k.txt"
    source.write_text("https://example.test/a.jpg -6.2 106.8\nhttps://example.test/b.jpg -7.5 110.4\n")
    import_capability("im2gps3k", source, tmp_path / "out.jsonl", image_root, "train", LICENSE)
    records = load_capability_manifest(tmp_path / "out.jsonl")
    assert records[0].extensions["capability"]["gps"] == [-6.2, 106.8]
    assert records[1].extensions["capability"]["source_url"].endswith("b.jpg")


def test_osv5m_capability_import_from_csv(tmp_path, image_root):
    add_image(image_root, "osv/img1.jpg")
    source = tmp_path / "osv5m.csv"
    source.write_text("image,lat,lon,city,country\nosv/img1.jpg,-6.9,107.6,Bandung,ID\n")
    import_capability("osv5m", source, tmp_path / "out.jsonl", image_root, "train", LICENSE)
    records = load_capability_manifest(tmp_path / "out.jsonl")
    capability = records[0].extensions["capability"]
    assert capability["gps"] == [-6.9, 107.6] and capability["city"] == "Bandung"


def test_vggface2_capability_import_identity_classes(tmp_path, image_root):
    add_image(image_root, "n0001/a.jpg")
    add_image(image_root, "n0001/b.jpg")
    source = tmp_path / "list.csv"
    source.write_text("n0001,a.jpg\nn0001,b.jpg\n")
    import_capability("vggface2", source, tmp_path / "out.jsonl", image_root, "train", LICENSE)
    records = load_capability_manifest(tmp_path / "out.jsonl")
    assert {r.extensions["capability"]["identity_class"] for r in records} == {"n0001"}
    assert all("consent" in r.extensions["capability"]["usage_note"] for r in records)


def test_yfcc100m_capability_import_capture_dates(tmp_path, image_root):
    add_image(image_root, "1234.jpg")
    source = tmp_path / "yfcc.tsv"
    fields = [
        "1234",
        "user1",
        "2014-05-06 10:11:12.0",
        "2014-06-01 00:00:00.0",
        "Canon EOS",
        "sunset",
        "beach",
        "sea",
        "",
        "106.8",
        "-6.2",
        "16",
        "1",
        "https://example.test/p/1234",
    ]
    source.write_text("\t".join(fields) + "\n")
    import_capability("yfcc100m", source, tmp_path / "out.jsonl", image_root, "train", LICENSE)
    records = load_capability_manifest(tmp_path / "out.jsonl")
    capability = records[0].extensions["capability"]
    assert capability["kind"] == "photo-date-validation"
    assert capability["capture_time"].startswith("2014-05-06")
    assert capability["gps"] == [-6.2, 106.8]


def test_importers_reject_path_escape(tmp_path, image_root):
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps({"caption": "A dog", "image": "../../escape.jpg", "label": "entailment"}))
    with pytest.raises(ValueError, match="escapes image root"):
        import_caption_dataset("snli-ve", source, tmp_path / "out.jsonl", image_root, "train", LICENSE)


def test_capability_manifest_rejects_split_leakage(tmp_path, image_root):
    add_image(image_root, "0.jpg")
    add_image(image_root, "1.jpg")
    # Identical bytes under two names: same image hash across splits must leak.
    (image_root / "1.jpg").write_bytes((image_root / "0.jpg").read_bytes())
    # Real workflow: rows are partitioned into separate input files per split.
    (tmp_path / "train-rows.txt").write_text("https://a.example/0.jpg -6.2 106.8\n")
    (tmp_path / "test-rows.txt").write_text("https://a.example/1.jpg -6.2 106.9\n")
    import_capability(
        "im2gps3k", tmp_path / "train-rows.txt", tmp_path / "train.jsonl", image_root, "train", LICENSE
    )
    import_capability(
        "im2gps3k", tmp_path / "test-rows.txt", tmp_path / "test.jsonl", image_root, "test", LICENSE
    )
    combined = Path(tmp_path / "combined.jsonl")
    combined.write_text((tmp_path / "train.jsonl").read_text() + (tmp_path / "test.jsonl").read_text())
    with pytest.raises(ValueError, match="Split leakage"):
        load_capability_manifest(combined)


def test_source_lists_are_disjoint_and_complete():
    assert not set(CAPTION_SOURCES) & set(CAPABILITY_SOURCES)
    assert set(CAPTION_SOURCES) | set(CAPABILITY_SOURCES) == {
        "snli-ve",
        "countbench",
        "refcoco",
        "flickr30k-entities",
        "visual-genome",
        "sa1b",
        "fsc-147",
        "im2gps3k",
        "osv5m",
        "vggface2",
        "yfcc100m",
    }
