# deep-dream-keras

Deep Dream implementation inspired from official [keras implementaion](https://keras.io/examples/generative/deep_dream/).

Script to keep dreaming in a loop and creating a video from the frames.

## Example

- On one image

    ```python
    python src/dream.py \
      --src_img inputs/flowers_pixel.jpg \
      --result_img results/flowers_pixel_dream.jpg \
      --downscale_factor 1.5 
    ```

- Keep Dreaming
  
  ```python
  python src/keep_dreaming.py \
    --src_img inputs/aus.jpg \
    --directory dream_seq \
    --iterations 100 \
    --overwrite_save_dir false
  ```

  ```python

  python src/create_video.py \
    --frames_dir dream_seq \
    --fps 60 \
    --output_file results/video.avi

  ```

  ```python
    OUTPUT:

    100%|█████████████████████████████████████████████████| 7/7 [00:00<00:00, 218.68it/s] 
    video saved: results/video.avi 
   ```
