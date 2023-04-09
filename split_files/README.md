# MovieCLIP Dataset

## Complete Taxonomy

* The entire list of 179 visual scene labels in MovieCLIP dataset can be found in the file **MovieCLIP_taxonomy_split.txt**.

## Raw videos 

* Download the original videos by requesting access to the [**Condensed Movies Dataset**](https://github.com/m-bain/CondensedMovies). Our video tagging was peformed on the videos present in **Condensed Movies** dataset. We do not own the raw videos.

## CLIP tags 

* The complete list of CLIP tags for the shots in the MovieCLIP dataset can be downloaded from this [**Drive Link**](https://drive.google.com/file/d/15EhA0BT3IF0EuLP1yXr5nn5ad9soxxox/view?usp=share_link)

* Load the CLIP tags using the following code snippet:

    ```python
    import json
    with open('movieCLIP_dataset.json', 'r') as f:
        movieCLIP_tags = json.load(f)
    ```
* **movieCLIP_tags** is a dictionary with keys as the video names (youtube ids in **Condensed Movies**) and values as a list of CLIP tags for each shot in the video:

    ```python
    "qM8jk56Vj9Y":
        "qM8jk56Vj9Y-Scene-018.mp4": {
                "start_frame": 1059.0,
                "end_frame": 1137.0,
                "start_time": 44.169,
                "end_time": 47.422,
                "labels": {
                    "banquet": 0.7861328125,
                    "dining room": 0.07110595703125,
                    "restaurant": 0.028594970703125,
                    "penthouse": 0.01611328125,
                    "salon": 0.01186370849609375
                }
            },
    "B-tq7mbTvrA":
        "B-tq7mbTvrA-Scene-003.mp4": {
            "start_frame": 54.0,
            "end_frame": 74.0,
            "start_time": 2.252,
            "end_time": 3.086,
            "labels": {
                "batting cage": 0.479248046875,
                "locker room": 0.160400390625,
                "baseball field": 0.11248779296875,
                "stadium": 0.0601806640625,
                "bowling alley": 0.040496826171875
            }
        },
    "Ld2g77JckSk":
        "Ld2g77JckSk-Scene-018.mp4": {
            "start_frame": 974.0,
            "end_frame": 1031.0,
            "start_time": 40.627,
            "end_time": 43.004,
            "labels": {
                "animal shelter": 0.640625,
                "zoo": 0.07684326171875,
                "farm": 0.04071044921875,
                "fair": 0.0256500244140625,
                "suburban": 0.0123748779296875
            }
        }
    ```

    For example, to access the CLIP tags associated with the shot```qM8jk56Vj9Y-Scene-018.mp4``` present in the video id ```qM8jk56Vj9Y-Scene-018.mp4```, use ```movieCLIP_tags['qM8jk56Vj9Y']['qM8jk56Vj9Y-Scene-018.mp4']['labels']```. This gives CLIP tags with their corresponding ```CLIPScene``` scores for the shot.


## Label distribution 

* Out of the 179 visual scene labels present in **MovieCLIP_taxonomy_split.txt**, we use 150 visual scene labels after thresholding on the top-1 CLIPScene score (>=0.4) and top-k(k=2 to 5) CLIPScene scores (>=0.1). the list of 150 visual scene labels can be found in the file **label_2_ind_multi_label_thresh_0_4_0_1_150_label_map.pkl**. Sample mapping is shown as below

```python
{'courtroom': 0,'police station': 1,'mountain': 2,'swamp': 3,
 'train': 4,'corridor': 5,'baseball field': 6,'garage': 7,'bakery': 8,'stairs': 9, 'pool': 10,'road': 11,'park': 12,
'church': 13,'desert': 14,...}
```







