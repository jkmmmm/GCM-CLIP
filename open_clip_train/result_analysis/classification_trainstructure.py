
class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, disease_category, disease_location, config, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.disease_category = df[disease_category].tolist()
        self.disease_location = df[disease_location].tolist()
        self.prompt_templates = config['student_prompt_template']
        self.category = df["category"].tolist()
        self.location = df["location"].tolist()
        self.health_label_to_idx = {v: int(k)-1 for k, v in config['category'].items()}
        self.location_label_to_idx = {v: int(k)-1 for k, v in config['location'].items()}
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer
        

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        category = str(self.category[idx])
        location = str(self.location[idx])
        # random_number = random.randint(0, len(self.prompt_templates)-1)
        # DiseaseCategory = str(self.disease_category[idx])
        # DiseaseLocation = str(self.disease_location[idx])
        # prompt = self.prompt_templates[random_number].format(f=DiseaseCategory, b=DiseaseLocation)
        # texts = self.tokenize([prompt])[0]
        # 健康状况标签
        health_name = self.category[idx] 
        health_label = torch.zeros(len(self.health_label_to_idx), dtype=torch.float)
        if health_name in self.health_label_to_idx:
            health_label[self.health_label_to_idx[health_name]] = 1.0
                
        # 身体部位标签
        location_name = self.location[idx]
        location_label = torch.zeros(len(self.location_label_to_idx), dtype=torch.float)
        if location_name in self.location_label_to_idx:
            location_label[self.location_label_to_idx[location_name]] = 1.0
        
        return images, texts, health_label, location_label

