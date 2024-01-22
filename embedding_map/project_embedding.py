from sklearn.manifold import TSNE

def main(embedding_dataset):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(embedding_dataset)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()

if __name__ == "__main__":
    main()
