import csv

def readcsv(fileurl):
    # Ouvrir le fichier CSV
    with open(fileurl, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')

        # Initialiser la variable pour stocker le contenu
        contenu = ''

        # Parcourir les lignes du fichier
        for row in csvreader:
            # Supprimer la partie de temps
            row.pop(1)
            row.pop(1)
            # Ajouter la ligne à la variable de contenu
            contenu += ' '.join(row) + ' '

    # Afficher le contenu

    contenu = contenu.replace("SPEAKER","\n SPEAKER")
    return contenu

def t():
    a=1
    b=2
    return a,b


print(t()[1])