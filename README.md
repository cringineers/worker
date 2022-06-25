# Worker service

Сервис выполняющий вычисления связанные с предсказанием нейронной сети CLIP.

При наличии видеокарты с cuda будет использовать ее ресурсы, иначе будет работать только на cpu. 

## Routes
    Post: 
        /features_tag - возвращает выход энкодера текта для заданного тега на английском языке:
          headers: {"content-type": "application/json"}
          body: {"text": <tag_text>}
          response example: {"features": [0.5, ... 0.234]}
          success code: 200 
        
        /features_image - возвращает выход энкодера текта для заданного изображения:
          body: <binary_image_file>
          response example: {"features": [0.5, ... 0.234]}
          success code: 200 
        
        /predict_image - возвращает предсказание для заданных изображений на группе тегов:
          headers: {"content-type": "application/json"}
          body: {
            "images": [
                {"id": <image_id_1>, "latent_space": <image_features_1>},
                {"id": <image_id_2>, "latent_space": <image_features_2>},
                ...
                {"id": <image_id_n>, "latent_space": <image_features_n>}
            ],
            "tag_groups": [
                {
                    "id": <group_id_1>, 
                    "name": <group_name_1>,
                    "tags": [
                        {"id": <tag_id_1>, "name": <tag_name_1>, "latent_space": <tag_latent_space_1>},
                        ...
                        {"id": <tag_id_k>, "name": <tag_name_k>, "latent_space": <tag_latent_space_k>}
                    ]
                },
                ...
                {
                    "id": <group_id_n>, 
                    "name": <group_name_n>,
                    "tags": [
                        {"id": <tag_id_1>, "name": <tag_name_1>, "latent_space": <tag_latent_space_1>},
                        ...
                        {"id": <tag_id_m>, "name": <tag_name_m>, "latent_space": <tag_latent_space_m>}
                    ]
                }
             ]
          }
          request example: {
              <group_id_1>: [
                  {"image_id": <image_id>, "tag_id": <tag_id>}
                  ...
              ],
              ...
          }
          success code: 200 
