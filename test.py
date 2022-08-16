class Animal:
    def __init__(self, animal_list):
        self.animals_name = animal_list

    def __len__(self):
        return len(self.animals_name)
    #
    def __getitem__(self, index):
        return self.animals_name[index]

animals = Animal(["dog","cat","fish"])
for index,animal in enumerate(animals):
    print(index,animal)
