import hashlib
import os

CHECKSUM_MAP = {
    'annotations/labeled.json': '400d0efb8f2b50d6f41d0b2b4890025f',
    'annotations/test_a.json': '97cdc225a90efcb3a3a279f90a209d61',
    'annotations/unlabeled.json': '2c59fb2fdd2abce565f499fef9f8263d',
    'zip_feats/labeled.zip': '147dac81280075dc862fe45fe1475306',
    'zip_feats/test_a.zip': '9f30188a1f25ef971cf04e58b51ca999',
    'zip_feats/unlabeled.zip': '16c6cb4c3873dbbab918572727590b60',
}


def get_checksum(file_path):
    with open(file_path, 'rb') as f:
        md5 = hashlib.md5()
        block_size = 2 ** 20
        while True:
            data = f.read(block_size)
            if not data:
                break
            md5.update(data)
    checksum = md5.hexdigest()
    return checksum


def verify_data(data_dir):
    """
    to verify the integrity of the whole downloaded data directory
    :param data_dir: the downloaded data directory
    :return: files missing or failed in the verification
    """
    missing = []
    failed = []
    for file, checksum in CHECKSUM_MAP.items():
        file_path = os.path.join(data_dir, file)
        if not os.path.isfile(file_path):
            print(f'ERROR: file "{file}" is MISSING in "{data_dir}"')
            missing.append(file)
            continue
        if checksum != get_checksum(file_path):
            print(f'ERROR: checksum is inconsistent for file "{file}"')
            failed.append(file)
        else:
            print(f'checksum is consistent for file "{file}"')
    return missing, failed


if __name__ == '__main__':
    missing, failed = verify_data('./')
    if missing or failed:
        print(f'Verification FAILED, please retry downloading: {missing + failed}')
    else:
        print('Verification Passed!')
