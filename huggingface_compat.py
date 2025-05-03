"""
Compatibility layer for huggingface_hub
This provides backward compatibility for code that relies on cached_download
"""

from huggingface_hub import hf_hub_download

# Provide backward compatibility for cached_download
def cached_download(*args, **kwargs):
    """
    Backward compatibility wrapper for cached_download function
    
    This redirects to hf_hub_download which is the modern equivalent
    """
    # Convert legacy parameters if needed
    if 'cache_dir' in kwargs and 'local_dir' not in kwargs:
        kwargs['local_dir'] = kwargs.pop('cache_dir')
    
    if 'pretrained_model_name_or_path' in kwargs and 'repo_id' not in kwargs:
        kwargs['repo_id'] = kwargs.pop('pretrained_model_name_or_path')
    
    if 'filename' in kwargs and 'filename' not in kwargs:
        kwargs['filename'] = kwargs.pop('filename')
    
    # Call the modern function
    return hf_hub_download(*args, **kwargs)