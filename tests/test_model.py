import torch
import numpy as np
from podcnf.NFmodel import NormalizingFlow
from podcnf.Loader import LoadData

def test_flow_shapes():
    """
    Test to verify if the mdodel executes the forward step and the sampling
    returning the tensors with the correct dimensions
    """

    batch_size = 10
    dim_x = 4
    dim_y = 6

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NormalizingFlow(dim_x=dim_x, dim_y=dim_y, num_flows=4, device=device)
    
    sample_x = torch.randn(batch_size, dim_x, device=device)
    sample_y = torch.randn(batch_size, dim_y, device=device)

    # Log-prob
    log_p = model(sample_x, sample_y)
    assert log_p.shape == (batch_size,), f"True shape {(batch_size,)}, obtained {log_px.shape}"
    
    # sample
    samples = model.sample(sample_x)
    assert samples.shape == (batch_size, dim_y), f"True shape {(batch_size, dim_y)}, obtained {samples.shape}"

def test_load_data_pipeline():
    """
    Test if the split and the standardization went well
    """
    total_samples = 60
    dim_mu = 4
    dim_c = 6
    batch_size = 10
    
    # random array
    mock_mu = np.random.randn(total_samples, dim_mu)
    mock_c = np.random.randn(total_samples, dim_c)
    
    n_train = 30
    n_val = 50
    
    # Test with StandardScaler = True
    train_l, val_l, test_l, mu_scale, c_scale = LoadData(
        mock_mu, mock_c, n_train, n_val, BATCH_SIZE=batch_size, norm_scaler=True
    )
    
    first_batch = next(iter(train_l))
    
    assert first_batch.shape == (batch_size, dim_mu + dim_c), f"Wrong shape: {first_batch.shape}"
    assert mu_scale is not None
    assert c_scale is not None