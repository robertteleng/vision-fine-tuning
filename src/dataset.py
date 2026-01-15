from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def get_dataset_stats():
    """Get statistics about the current dataset."""
    dataset_dir = DATA_DIR / "dataset"
    if not dataset_dir.exists():
        return "No dataset found at data/dataset/"

    stats = "## Dataset Statistics\n\n"

    for split in ['train', 'val']:
        images_dir = dataset_dir / split / 'images'
        labels_dir = dataset_dir / split / 'labels'

        if images_dir.exists():
            images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []

            # Count annotations
            total_annotations = 0
            for label_file in labels:
                content = label_file.read_text().strip()
                if content:
                    total_annotations += len(content.split('\n'))

            stats += f"### {split.capitalize()}\n"
            stats += f"- Images: {len(images)}\n"
            stats += f"- Labels: {len(labels)}\n"
            stats += f"- Total annotations: {total_annotations}\n"
            if len(images) > 0:
                stats += f"- Avg annotations/image: {total_annotations/len(images):.2f}\n"
            stats += "\n"

    return stats


def validate_dataset():
    """Validate dataset integrity."""
    dataset_dir = DATA_DIR / "dataset"
    if not dataset_dir.exists():
        return "No dataset found"

    issues = []

    for split in ['train', 'val']:
        images_dir = dataset_dir / split / 'images'
        labels_dir = dataset_dir / split / 'labels'

        if not images_dir.exists():
            issues.append(f"❌ {split}/images/ not found")
            continue

        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        # Check for missing labels
        missing_labels = []
        for img in images:
            label_path = labels_dir / f"{img.stem}.txt"
            if not label_path.exists():
                missing_labels.append(img.name)

        if missing_labels:
            issues.append(f"⚠️ {split}: {len(missing_labels)} images without labels")

        # Check label format
        invalid_labels = []
        for label_file in labels_dir.glob("*.txt") if labels_dir.exists() else []:
            content = label_file.read_text().strip()
            if not content:
                continue
            for line_num, line in enumerate(content.split('\n'), 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    invalid_labels.append(f"{label_file.name}:{line_num}")
                    continue
                try:
                    int(parts[0])  # class
                    for val in parts[1:5]:
                        v = float(val)
                        if not (0 <= v <= 1):
                            invalid_labels.append(f"{label_file.name}:{line_num} (out of range)")
                except ValueError:
                    invalid_labels.append(f"{label_file.name}:{line_num}")

        if invalid_labels:
            issues.append(f"❌ {split}: {len(invalid_labels)} invalid label lines")

    if not issues:
        return "✅ Dataset is valid! No issues found."

    return "## Validation Results\n\n" + "\n".join(issues)


def export_dataset(export_format: str, output_name: str):
    """Export dataset to different formats."""
    import shutil
    import zipfile
    from datetime import datetime

    dataset_dir = DATA_DIR / "dataset"
    if not dataset_dir.exists():
        return "No dataset found"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = output_name or f"dataset_export_{timestamp}"

    if export_format == "ZIP (YOLO format)":
        output_path = PROJECT_ROOT / "exports" / f"{output_name}.zip"
        output_path.parent.mkdir(exist_ok=True, parents=True) # Ensure exports dir exists

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in dataset_dir.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(dataset_dir)
                    zf.write(file, arcname)

        return f"✅ Exported to: `{output_path}`\n\nSize: {output_path.stat().st_size / 1024 / 1024:.2f} MB"

    elif export_format == "Copy to folder":
        output_path = PROJECT_ROOT / "exports" / output_name
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(dataset_dir, output_path)
        return f"✅ Copied to: `{output_path}`"

    return "Unknown export format"
