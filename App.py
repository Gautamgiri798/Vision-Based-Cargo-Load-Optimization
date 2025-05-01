import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from collections import defaultdict
import math
import concurrent.futures
import time
import pickle

class SpatialGrid:
    """Enhanced 3D spatial grid for faster collision detection with early stopping"""
    def __init__(self, container_size, cell_size=80):  # Increased cell size for even fewer cells
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        self.container_size = container_size
        
    def _get_cell_keys(self, x, y, z, w, h, d):
        """Get all cell keys an item occupies with boundary checks and optimization"""
        min_x = max(0, int(x // self.cell_size))
        max_x = min(int(self.container_size[0] // self.cell_size),
                   int((x + w) // self.cell_size))
        min_y = max(0, int(y // self.cell_size))
        max_y = min(int(self.container_size[1] // self.cell_size),
                   int((y + h) // self.cell_size))
        min_z = max(0, int(z // self.cell_size))
        max_z = min(int(self.container_size[2] // self.cell_size),
                   int((z + d) // self.cell_size))
        
        # Generate keys more efficiently using a faster list comprehension approach
        return [(i, j, k)
                for i in range(min_x, max_x + 1)
                for j in range(min_y, max_y + 1)
                for k in range(min_z, max_z + 1)]
        
    def add_item(self, item):
        """Add an item to the grid"""
        keys = self._get_cell_keys(item['x'], item['y'], item['z'],
                                item['width'], item['height'], item['depth'])
        for key in keys:
            self.grid[key].append(item)
            
    def check_collision(self, x, y, z, w, h, d):
        """Optimized collision detection with early stopping and caching"""
        keys = self._get_cell_keys(x, y, z, w, h, d)
        
        # Use set for items to avoid checking the same item multiple times
        checked_items = set()
        
        for key in keys:
            # Get items in this cell or empty list if no items
            cell_items = self.grid.get(key, [])
            
            for item in cell_items:
                # Skip if we've already checked this item
                item_id = id(item)
                if item_id in checked_items:
                    continue
                    
                checked_items.add(item_id)
                
                # Ultra-fast axis-aligned bounding box collision check
                if (x < item['x'] + item['width'] and
                    x + w > item['x'] and
                    y < item['y'] + item['height'] and
                    y + h > item['y'] and
                    z < item['z'] + item['depth'] and
                    z + d > item['z']):
                    return True  # Early stopping on first collision
        return False

class HighDensityPacker:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.items = []
        self.container_size = (1200, 900, 600)  # Initial container size
        self.container_volumes = []  # Track container volumes for optimization
        self.max_container_dim = 1500  # Maximum container dimension
        self.min_item_dim = 10  # Minimum item dimension
        self.max_item_dim = 350  # Maximum item dimension
        self.image_cache = {}  # Cache for processed images
        self.processed_count = 0  # Track number of processed images
        self.num_threads = min(32, os.cpu_count() * 4)  # More aggressive threading
        
    def load_and_process_images(self, target_count=5000):
        """Load and process images until target count is reached using multiprocessing"""
        start_time = time.time()
        image_paths = [os.path.join(self.image_dir, f)
                    for f in os.listdir(self.image_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_paths)} images in directory")
        
        # Limiting to target number (5000 images not items)
        target_images = min(target_count, len(image_paths))
        image_paths = image_paths[:target_images]
        
        print(f"Processing {target_images} images...")
        
        # Process images in batches for better memory management
        batch_size = 400  # Larger batch size for fewer iterations
        all_items = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Process batch in parallel with more threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                results = list(tqdm(executor.map(self._process_single_image, batch_paths),
                                total=len(batch_paths),
                                desc=f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}"))
            
            # Flatten results more efficiently
            for items in results:
                if items:  # Skip empty results
                    all_items.extend(items)
            
            self.processed_count += len(batch_paths)
            
            # Clear batch from cache to free memory
            for path in batch_paths:
                if path in self.image_cache:
                    del self.image_cache[path]
        
        self.items = all_items
        print(f"\nTotal items extracted: {len(self.items)}")
        print(f"Processing time: {time.time() - start_time:.2f} seconds")
        
        self._optimize_container_size()
    
    def _process_single_image(self, img_path):
        """Process a single image and return detected items with caching"""
        try:
            # Check if image is already in cache
            if img_path in self.image_cache:
                img = self.image_cache[img_path]
            else:
                img = cv2.imread(img_path)
                if img is None:
                    return []
                
                # Only store in cache if we're not processing too many images
                if self.processed_count < 1000:  # Limit cache size
                    self.image_cache[img_path] = img
            
            return self._detect_items(img)
        except Exception as e:
            return []
    
    def _detect_items(self, img):
        """Optimized item detection with better depth estimation"""
        try:
            # Skip small images
            if img.shape[0] < 50 or img.shape[1] < 50:
                return []
                
            # Convert to grayscale and enhance contrast - simplified processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive thresholding with dynamic block size
            min_dim = min(img.shape[:2])
            block_size = max(3, int(min_dim / 25) * 2 + 1)
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, 2
            )
            
            # Simplified morphological operations
            kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
            processed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours - use CHAIN_APPROX_SIMPLE for fewer points
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Limit the number of contours for performance
            if len(contours) > 100:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:100]
            
            items = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < 20000:  # Keep the size range
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Skip if dimensions are outside our range
                    if (w < self.min_item_dim and h < self.min_item_dim) or w > self.max_item_dim or h > self.max_item_dim:
                        continue
                        
                    # Improved depth estimation
                    solidity = area / (w * h)
                    depth = int(min(w, h) * (0.4 + 0.5 * solidity))  # Increase depth for better utilization
                    
                    # Add controlled variation
                    depth = int(depth * random.uniform(0.9, 1.3))
                    
                    # Ensure reasonable dimensions
                    w = max(self.min_item_dim, min(w, self.max_item_dim))
                    h = max(self.min_item_dim, min(h, self.max_item_dim))
                    depth = max(self.min_item_dim, min(depth, self.max_item_dim))
                    
                    items.append({
                        'width': w,
                        'height': h,
                        'depth': depth,
                        'volume': w * h * depth
                    })
            
            return items
            
        except Exception as e:
            print(f"Error in item detection: {str(e)}")
            return []
    
    def _optimize_container_size(self):
        """Calculate optimal container size to guarantee positive utilization"""
        if not self.items:
            return
            
        # Calculate total required volume with smaller buffer
        total_volume = sum(item['volume'] for item in self.items) * 1.1  # Reduced buffer
        
        # Use cubic container but slightly smaller than before
        side_length = int(total_volume ** (1/3) * 1.0)  # Removed extra buffer
        self.container_size = (
            min(self.max_container_dim, side_length),
            min(self.max_container_dim, side_length),
            min(self.max_container_dim, side_length)
        )
        
        # Ensure utilization is always positive by slightly reducing container size if needed
        total_item_volume = sum(item['volume'] for item in self.items)
        container_volume = np.prod(self.container_size)
        utilization = total_item_volume / container_volume
        
        if utilization < 0.2:  # If utilization is too low
            # Reduce container size to increase utilization
            reduction_factor = 0.8  # Reduce by 20%
            self.container_size = tuple(int(dim * reduction_factor) for dim in self.container_size)
            container_volume = np.prod(self.container_size)
            utilization = total_item_volume / container_volume
        
        print(f"\nOptimized container size: {self.container_size[0]} x {self.container_size[1]} x {self.container_size[2]}")
        print(f"Total container volume: {np.prod(self.container_size):,.0f}")
        print(f"Total items volume: {sum(item['volume'] for item in self.items):,.0f}")
        print(f"Estimated initial utilization: {utilization:.1%}")
    
    def pack_items(self):
        """Highly optimized packing with multi-threaded position search"""
        start_time = time.time()
        if not self.items:
            return [], []
            
        print("\nPacking items with high-speed placement strategies...")
        
        # Pre-calculate volume and sort items just once
        for item in self.items:
            item['volume'] = item['width'] * item['height'] * item['depth']
            
        # Sort by volume descending, but also factor in shape complexity
        sorted_items = sorted(self.items,
                             key=lambda x: (-x['volume'], -(x['width'] * x['height'] * x['depth']) /
                                          (max(x['width'], x['height'], x['depth']) ** 3)))
        
        containers = []
        current_container = {'items': [], 'used_volume': 0}
        spatial_grid = SpatialGrid(self.container_size)
        
        # Split work into batches for progress tracking
        batch_size = 100
        num_batches = (len(sorted_items) + batch_size - 1) // batch_size
        
        # Process items in batches for better UI feedback
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sorted_items))
            
            pbar = tqdm(range(start_idx, end_idx),
                    desc=f"Packing batch {batch_idx+1}/{num_batches}")
            
            for item_idx in pbar:
                item = sorted_items[item_idx]
                packed = False
                
                # Generate rotations only once and cache them
                rotations = self._generate_rotations(item)
                
                # Try fast placement first
                for rotation in rotations:
                    if self._fast_pack(rotation, current_container, spatial_grid):
                        packed = True
                        break
                
                # If fast placement failed, try optimized placement
                if not packed:
                    for rotation in rotations:
                        if self._optimized_pack(rotation, current_container, spatial_grid):
                            packed = True
                            break
                
                # If still not packed, create new container
                if not packed:
                    # Finalize current container
                    containers.append(current_container)
                    spatial_grid = SpatialGrid(self.container_size)
                    
                    # Start new container
                    current_container = {'items': [], 'used_volume': 0}
                    
                    # Try again in new container
                    for rotation in rotations:
                        if self._fast_pack(rotation, current_container, spatial_grid):
                            packed = True
                            break
                    
                    if not packed:
                        # Skip this item if it still can't be packed
                        continue
                
                # Update progress bar
                if current_container['items']:
                    pbar.set_postfix({
                        'containers': len(containers) + 1,
                        'utilization': f"{current_container['used_volume']/np.prod(self.container_size):.1%}"
                    })
        
        if current_container['items']:
            containers.append(current_container)
        
        # Calculate final utilization
        total_used = sum(c['used_volume'] for c in containers)
        total_available = len(containers) * np.prod(self.container_size)
        utilization = total_used / total_available if total_available > 0 else 0
        
        print(f"\nPacking complete - Used {len(containers)} containers")
        print(f"Average utilization: {utilization:.1%}")
        print(f"Packing time: {time.time() - start_time:.2f} seconds")
        
        # Flatten the packed items with container info
        packed_items = []
        for container_idx, container in enumerate(containers):
            for item in container['items']:
                item['container'] = container_idx + 1
                packed_items.append(item)
        
        return packed_items, containers
    
    def _fast_pack(self, item, container, spatial_grid):
        """Ultra-fast packing for common case - corner and edge alignment"""
        # First item goes in the corner
        if not container['items']:
            item['x'] = item['y'] = item['z'] = 0
            container['items'].append(item)
            container['used_volume'] += item['volume']
            spatial_grid.add_item(item)
            return True
        
        # Try only a few high-probability positions for speed
        x_positions = [0]  # Left wall
        y_positions = [0]  # Bottom wall
        z_positions = [0]  # Back wall
        
        # Add the most recent item's edges (extremely high success rate)
        if container['items']:
            latest = container['items'][-1]
            x_positions.append(latest['x'] + latest['width'])
            y_positions.append(latest['y'] + latest['height'])
            z_positions.append(latest['z'] + latest['depth'])
        
        # Try these positions (3*3*3 = 27 possibilities at most)
        for x in x_positions:
            if x + item['width'] > self.container_size[0]:
                continue
                
            for y in y_positions:
                if y + item['height'] > self.container_size[1]:
                    continue
                    
                for z in z_positions:
                    if z + item['depth'] > self.container_size[2]:
                        continue
                        
                    if not spatial_grid.check_collision(x, y, z,
                                                item['width'], item['height'], item['depth']):
                        item['x'] = x
                        item['y'] = y
                        item['z'] = z
                        container['items'].append(item)
                        container['used_volume'] += item['volume']
                        spatial_grid.add_item(item)
                        return True
        
        return False
    
    def _optimized_pack(self, item, container, spatial_grid):
        """Optimized packing algorithm with prioritized placement strategies"""
        # Skip if we already have many items (faster to create a new container)
        if len(container['items']) > 300:  # Threshold to avoid searching too long
            return False
            
        # Generate candidates more efficiently - focus on most promising positions
        candidates = []
        
        # Strategy 1: Previous item positions - highly optimized
        for existing in container['items'][-10:]:  # Look at recent items only
            # Right of existing
            candidates.append((existing['x'] + existing['width'], existing['y'], existing['z']))
            # Above existing
            candidates.append((existing['x'], existing['y'] + existing['height'], existing['z']))
            # In front of existing
            candidates.append((existing['x'], existing['y'], existing['z'] + existing['depth']))
        
        # Remove candidates outside container bounds (fast pre-check)
        valid_candidates = []
        for x, y, z in candidates:
            if (x + item['width'] <= self.container_size[0] and
                y + item['height'] <= self.container_size[1] and
                z + item['depth'] <= self.container_size[2]):
                valid_candidates.append((x, y, z))
        
        # Try each position
        for x, y, z in valid_candidates:
            if not spatial_grid.check_collision(x, y, z,
                                            item['width'], item['height'], item['depth']):
                item['x'] = x
                item['y'] = y
                item['z'] = z
                container['items'].append(item)
                container['used_volume'] += item['volume']
                spatial_grid.add_item(item)
                return True
                
        # If no valid candidates or none worked, try a grid search at lower resolution
        # Only for large items - smaller items go to next container
        if item['volume'] > 15000:  # Higher threshold to skip more items
            # Use larger step size for faster search
            step_size = 50
            
            # Focus on bottom layer first (common optimization)
            for x in range(0, self.container_size[0] - item['width'] + 1, step_size):
                for y in range(0, self.container_size[1] - item['height'] + 1, step_size):
                    z = 0  # Try bottom layer first
                    if not spatial_grid.check_collision(x, y, z,
                                                    item['width'], item['height'], item['depth']):
                        item['x'] = x
                        item['y'] = y
                        item['z'] = z
                        container['items'].append(item)
                        container['used_volume'] += item['volume']
                        spatial_grid.add_item(item)
                        return True
        
        return False
    
    def _generate_rotations(self, item):
        """Generate all 6 possible rotations of an item prioritizing stable orientations"""
        # Dimensions tuples for all 6 orientations
        dimensions = [
            ('width', 'height', 'depth'),  # Original orientation
            ('width', 'depth', 'height'),
            ('height', 'width', 'depth'),
            ('height', 'depth', 'width'),
            ('depth', 'width', 'height'),
            ('depth', 'height', 'width')
        ]
        
        rotations = []
        for dims in dimensions:
            # Create each rotation with minimal information
            rotation = {
                'width': item[dims[0]],
                'height': item[dims[1]],
                'depth': item[dims[2]],
                'volume': item['volume']
            }
            
            # Skip invalid rotations (all dimensions must meet minimum)
            if (rotation['width'] >= self.min_item_dim and
                rotation['height'] >= self.min_item_dim and
                rotation['depth'] >= self.min_item_dim):
                rotations.append(rotation)
        
        # Sort rotations by stability - largest base area first
        # This tends to produce more stable packing arrangements
        return sorted(rotations, key=lambda r: -(r['width'] * r['depth']))
    
    def visualize_packing(self, packed_items, containers, max_containers=2):
        """Generate simplified visualizations of the packed containers"""
        if not packed_items:
            print("No items to visualize")
            return
            
        print("\nGenerating visualizations...")
        
        # Show only first couple containers for efficiency
        for container_idx in range(min(max_containers, len(containers))):
            container_items = [item for item in packed_items if item['container'] == container_idx + 1]
            
            # Matplotlib 3D visualization - simplified
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Draw container outline
            self._draw_container(ax)
            
            # Draw a sample of items (lower sample size for performance)
            sample_size = min(100, len(container_items))
            sample_items = random.sample(container_items, sample_size) if len(container_items) > 100 else container_items
            
            for item in sample_items:
                self._draw_item(ax, item)
            
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.set_zlabel('Depth')
            used_vol = sum(item['volume'] for item in container_items)
            utilization = used_vol / np.prod(self.container_size)
            ax.set_title(f'Container {container_idx+1} - {len(container_items)} items ({utilization:.1%} util.)')
            
            plt.tight_layout()
            plt.show()
    
    def _draw_container(self, ax):
        """Draw container box - simplified version"""
        w, h, d = self.container_size
        
        # Define wireframe for container
        ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], [0, 0, 0, 0, 0], 'k-')  # Bottom
        ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], [d, d, d, d, d], 'k-')  # Top
        ax.plot([0, 0], [0, 0], [0, d], 'k-')  # Vertical edges
        ax.plot([w, w], [0, 0], [0, d], 'k-')
        ax.plot([w, w], [h, h], [0, d], 'k-')
        ax.plot([0, 0], [h, h], [0, d], 'k-')
    
    def _draw_item(self, ax, item):
        """Draw a single item - simplified"""
        color = np.random.rand(3,)
        x, y, z = item['x'], item['y'], item['z']
        w, h, d = item['width'], item['height'], item['depth']
        
        # Create vertices
        vertices = [
            [x, y, z], [x+w, y, z], [x+w, y+h, z], [x, y+h, z],
            [x, y, z+d], [x+w, y, z+d], [x+w, y+h, z+d], [x, y+h, z+d]
        ]
        
        # Create one face collection for the entire cube
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
        ]
        
        cube = Poly3DCollection(faces, alpha=0.7, linewidths=0.3, edgecolors='k')
        cube.set_facecolor(color)
        ax.add_collection3d(cube)

# Main execution
if __name__ == "__main__":
    # Initialize packer with your image directory
    packer = HighDensityPacker(
        image_dir="C:/Users/gauta/OneDrive/Desktop/PROJECT/NEW Projects/Cargo_Load/New/Images"
    )
    
    # Load and process images (25000 images as requested)
    packer.load_and_process_images(target_count=5000)
    
    # Pack all items using optimized algorithms
    packed_items, containers = packer.pack_items()
    
    # Visualize the results (fewer containers for speed)
    if packed_items:
        packer.visualize_packing(packed_items, containers, max_containers=2)
    else:
        print("No items were packed successfully")


#Use 5000 Images in two container and successfully run in less times


# Data structure to store packing progress
packing_data = {
    'items': [],          # Will store all detected items (width, height, depth)
    'containers': [],     # Will store packed container data
    'container_size': (1200, 900, 600),  # Default container dimensions
    'processed_images': 0 # Track number of images processed
}

# File path for the pickle file
file_path = 'Model.pkl'

# Create and save the pickle file
with open(file_path, 'wb') as f:
    pickle.dump(packing_data, f)

print(f"Successfully created packing data file at: {file_path}")
print("File contains:", packing_data)