from tqdm import tqdm
import tarfile
import pickle
import os
from multiprocessing import Pool, cpu_count

# Function to process a batch of tar members
def process_members(members):
    tar_index_local = {}
    for member in members:
        if member.name.endswith('.jpg') and member.size > 5120:
            tar_index_local[member.name.split('/')[-1]] = member
    return tar_index_local

# Main function to parallelize tar indexing
def build_tar_index_parallel(tar_path, output_path, num_workers=None):
    tar_index = {}
    num_workers = num_workers or cpu_count()  # Use all CPU cores if not specified

    with tarfile.open(tar_path) as tar_obj:
        members = tar_obj.getmembers()
        print(f"Total members: {len(members)}")
        
        # Split tar members into chunks for each worker
        chunk_size = len(members) // num_workers
        member_chunks = [members[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
        if len(members) % num_workers:
            member_chunks[-1].extend(members[num_workers * chunk_size:])  # Add remaining members

        # Use multiprocessing Pool to process chunks in parallel
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_members, member_chunks), total=num_workers, desc="Building Index"))

        # Combine results from all workers
        for result in results:
            tar_index.update(result)

    # Save the combined index to a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(tar_index, f)
    print("Tar index building success!")

# Paths
tar_path = os.path.join('./data/', 'mp-16-images.tar')
output_path = os.path.join('./data/', 'tar_index.pkl')

# Run the function
if __name__ == '__main__':
    build_tar_index_parallel(tar_path, output_path, num_workers=8)  # Adjust workers as needed

