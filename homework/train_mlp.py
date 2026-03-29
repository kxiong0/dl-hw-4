import torch
import torch.utils.tensorboard as tb

from datetime import datetime
from pathlib import Path
from models import load_model, save_model
from metrics import PlannerMetric
from datasets.road_dataset import load_data


def train(exp_dir: str = "mlp_logs", num_epoch=10, batch_size=128, lr=0.05):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using mps")
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"mlp_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model("mlp_planner", with_weights=False)
    model.to(device)
    model.train()

    train_data = load_data(
        "../drive_data/train",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
        transform_pipeline="state_only",
    )
    val_data = load_data("../drive_data/val", shuffle=False)

    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    train_metric_computer = PlannerMetric()
    test_metric_computer = PlannerMetric()

    # training loop
    for epoch in range(num_epoch):
        print(f"{epoch=}")
        train_metric_computer.reset()
        test_metric_computer.reset()

        model.train()
        for entry in train_data:
            track_left = entry["track_left"].to(device)
            track_right = entry["track_right"].to(device)
            waypoints = entry["waypoints"].to(device)
            waypoints_mask = entry["waypoints_mask"].to(device)
            
            out = model(track_left, track_right)
            optimizer.zero_grad()
            
            # don't calculate loss on masked out waypoints
            waypoints_masked = waypoints * waypoints_mask.unsqueeze(-1)
            out_masked = out * waypoints_mask.unsqueeze(-1)
            loss_val = loss_func(out_masked, waypoints_masked)
            loss_val.backward()
            
            logger.add_scalar("train_loss", loss_val, global_step)

            train_metric_computer.add(out, waypoints, waypoints_mask)
            global_step += 1
            optimizer.step()

        # disable gradient computation and switch to evaluation mode
        model.eval()
        with torch.inference_mode():
            for entry in val_data:
                track_left = entry["track_left"].to(device)
                track_right = entry["track_right"].to(device)
                waypoints = entry["waypoints"].to(device)
                waypoints_mask = entry["waypoints_mask"].to(device)

                out = model(track_left, track_right)
                test_metric_computer.add(out, waypoints, waypoints_mask)

        # log average train and val accuracy to tensorboard
        logger.add_scalar("train_lateral_error", train_metric_computer.compute()["lateral_error"], global_step)
        logger.add_scalar("train_longitudinal_error", train_metric_computer.compute()["longitudinal_error"], global_step)
        logger.add_scalar("val_lateral_error", test_metric_computer.compute()["lateral_error"], global_step)
        logger.add_scalar("val_longitudinal_error", test_metric_computer.compute()["longitudinal_error"], global_step)

    save_model(model)

    # # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / "model.th")
    print(f"Model saved to {log_dir / 'model.th'}")


if __name__ == "__main__":
    train()
