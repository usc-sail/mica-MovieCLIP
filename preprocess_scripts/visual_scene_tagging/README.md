## Visual scnene tagging

* Use the file `../../split_files/MovieCLIP_taxonomy_split.txt` to run the script `visual_scene_tagging.py` to generate the visual scene tags for shots in `source_folder`.

```bash
CUDA_VISIBLE_DEVICES=0 python clip_scene_tagging.py --label_file ./../split_files/MovieCLIP_taxonomy_split.txt --source_folder <base folder containing the shots subfolder> --output_folder <output folder path containing the json file with nested dictionary containing visual scene classes>
```

