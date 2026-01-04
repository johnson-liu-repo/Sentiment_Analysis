if __name__ == "__main__":
    import os
    import sys
    import argparse
    import numpy as np
    import random
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    random.seed(1994)

    try:
        from sentiment_analysis.embeddings.pretrained_embeddings import (
            build_tfidf_and_vocab,
            load_pretrained_vectors,
            tfidf_weighted_average_embeddings,
            save_vectorized_comments
        )
    except Exception:
        from sentiment_analysis.embeddings.pretrained_embeddings import (
            build_tfidf_and_vocab,
            load_pretrained_vectors,
            tfidf_weighted_average_embeddings,
            save_vectorized_comments
        )

    parser = argparse.ArgumentParser(description="Sarcasm/Sentiment Analysis with pre-trained embeddings")
    parser.add_argument("--part", choices={"preprocess", "vectorize_pretrained", "train_fnn"}, required=True)

    parser.add_argument("--embeddings", default="glove-wiki-gigaword-300",
                        help="Pre-trained set to use (default: glove-wiki-gigaword-300 via gensim).")
    parser.add_argument("--glove_path", default="data/embeddings/glove.6B.300d.txt",
                        help="Path to local GloVe .txt (used if gensim download is unavailable).")
    parser.add_argument("--embedding_dim", type=int, default=300)

    # FNN training args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--num_workers", type=int, default=0)

    # NEW: checkpointing / resume
    parser.add_argument("--checkpoint_every", type=int, default=1,
                        help="Save a checkpoint every N epochs (default: 1). Set 0 to disable periodic checkpoints.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the most recent checkpoint if available.")
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Explicit path to a checkpoint to resume from.")

    args = parser.parse_args()

    preprocess_dir = 'data/training_data/test_training_02/preprocessing/'
    pretrained_dir = 'data/training_data/test_training_02/pretrained/'
    os.makedirs(pretrained_dir, exist_ok=True)

    if args.part == 'preprocess':
        raise SystemExit(
            "Preprocess step is unchanged. Use your existing pipeline to create "
            f"{preprocess_dir}text.npy and {preprocess_dir}labels.npy."
        )

    elif args.part == 'vectorize_pretrained':
        text_path = os.path.join(preprocess_dir, 'text.npy')
        labels_path = os.path.join(preprocess_dir, 'labels.npy')
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Missing file: {text_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Missing file: {labels_path}")

        print(f"Loading texts from {text_path}")
        texts = np.load(text_path, allow_pickle=True)
        print(f"Loading labels from {labels_path}")
        labels = np.load(labels_path, allow_pickle=True)

        print("Fitting TF-IDF and extracting vocabulary from corpus...")
        vectorizer, vocab = build_tfidf_and_vocab(texts, lowercase=True)

        print(f"Loading pre-trained embeddings: {args.embeddings} (or {args.glove_path} if offline)...")
        kv = load_pretrained_vectors(
            prefer=args.embeddings,
            glove_path=args.glove_path,
            restrict_vocab=vocab,
            expected_dim=args.embedding_dim
        )

        print("Vectorizing comments with TF-IDF–weighted average embeddings...")
        X = tfidf_weighted_average_embeddings(texts, vectorizer, kv, args.embedding_dim)

        out_vecs = os.path.join(pretrained_dir, 'vectorized_comments.npy')
        out_lbls = os.path.join(pretrained_dir, 'labels.npy')
        save_vectorized_comments(X, labels, out_vecs, out_lbls)
        print(f"Saved: {out_vecs} and {out_lbls}")

    elif args.part == 'train_fnn':
        try:
            from sentiment_analysis.models.feedforward_neural_network import custom_fnn
        except Exception:
            from sentiment_analysis.models.feedforward_neural_network import custom_fnn

        vecs_path = os.path.join(pretrained_dir, 'vectorized_comments.npy')
        labels_path = os.path.join(pretrained_dir, 'labels.npy')
        if not os.path.exists(vecs_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Expected vectorized comments and labels at {vecs_path} and {labels_path}. "
                "Run with --part vectorize_pretrained first."
            )

        print("Loading vectorized comments and corresponding labels...")
        X = np.load(vecs_path, allow_pickle=True)
        y = np.load(labels_path, allow_pickle=True)

        print(f"X shape: {X.shape}; y shape: {y.shape}")
        num_zeros = int((y == 0).sum())
        num_ones = int((y == 1).sum())
        print(f"Label balance — 0s: {num_zeros}, 1s: {num_ones}")

        print("Training the feedforward neural network (pre-trained embeddings)...")
        custom_fnn(
            X, y,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_workers=args.num_workers,
            seed=args.seed,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            resume_path=args.resume_path
        )
