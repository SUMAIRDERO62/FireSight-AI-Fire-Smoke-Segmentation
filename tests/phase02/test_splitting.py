from firesight.data.splitting import stable_bucket, assign_train_val

def test_stable_bucket_reproducible():
    assert stable_bucket("image-hash",42) == stable_bucket("image-hash",42)

def test_seed_changes_assignment_bucket():
    assert stable_bucket("image-hash",42) != stable_bucket("image-hash",7)

def test_assign_only_train_or_val():
    assert assign_train_val("abc",0.15,42) in {"train","val"}
