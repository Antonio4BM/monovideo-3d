import argparse
from pipeline.extract_frames import extract_frames
from pipeline.colmap_features_extractor import features_extractor
from pipeline.features_matcher import features_matcher
from pipeline.colmap_mapper import colmap_mapper
from pipeline.colmap_reconstructor import reconstructor
from pipeline.visualize import visualize
from pipeline.image_undistorter import image_undistorter
from pipeline.patch_match_stereo import patch_match_stereo
from pipeline.stereo_fusion import stereo_fusion

def build_parser():
    parser = argparse.ArgumentParser(description="MonoVideo3D pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_frames_parser = subparsers.add_parser(
        "extract-frames",
        help="Extract frames from a video"
    )
    extract_frames_parser.add_argument("--video_path", type=str, required=True)
    extract_frames_parser.add_argument("--output_path", type=str, required=True)
    extract_frames_parser.add_argument("--strides", type=int, default=20)

    colmap_features_extractor_parser = subparsers.add_parser(
        "colmap-features",
        help="Extract features from a video"
    )
    colmap_features_extractor_parser.add_argument("--images_path", type=str, required=True)

    colmap_feature_matcher_parser = subparsers.add_parser(
        "colmap-matches",
        help="Match features between images"
    )
    colmap_feature_matcher_parser.add_argument("--images_path", type=str, required=True)

    colmap_mapper_parser = subparsers.add_parser(
        "colmap-mapper",
        help="Map features between images"
    )
    colmap_mapper_parser.add_argument("--images_path", type=str, required=True)

    colmap_reconstructor_parser = subparsers.add_parser(
        "colmap-reconstructor",
        help="Reconstruct 3D model from images"
    )
    colmap_reconstructor_parser.add_argument("--images_path", type=str, required=True)

    colamp_visualizer_parser = subparsers.add_parser(
        "colmap-visualizer",
        help="Visualize 3D model"
    )

    colamp_visualizer_parser.add_argument("--images_path", type=str, required=True)
    colamp_visualizer_parser.add_argument("--filename", type=str, required=True)

    colmap_fusion_parser = subparsers.add_parser(
        "colmap-fusion",
        help="Fuse stereo images"
    )
    colmap_fusion_parser.add_argument("--images_path", type=str, required=True)

    colmap_dense_parser = subparsers.add_parser(
        "colmap-dense",
        help="Create dense point cloud"
    )
    colmap_dense_parser.add_argument("--images_path", type=str, required=True)

    colmap_sparse_parser = subparsers.add_parser(
        "colmap-sparse",
        help="Create sparse point cloud"
    )

    colmap_sparse_parser.add_argument("--video_path", type=str, required=True)
    colmap_sparse_parser.add_argument("--output_path", type=str, required=True)
    colmap_sparse_parser.add_argument("--strides", type=int, default=20)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "extract-frames":
        extract_frames(args.video_path, args.output_path, args.strides)
    elif args.command == "colmap-features":
        features_extractor(args.images_path)
    elif args.command == "colmap-matches":
        features_matcher(args.images_path)
    elif args.command == "colmap-mapper":
        colmap_mapper(args.images_path)
    elif args.command == "colmap-reconstructor":
        reconstructor(args.images_path)
    elif args.command == "colmap-visualizer":
        visualize(args.images_path, args.filename)
    elif args.command == "colmap-fusion":
        stereo_fusion(args.images_path)
    elif args.command == "colmap-sparse":
        extract_frames(args.video_path, args.output_path, args.strides)
        features_extractor(args.output_path)
        features_matcher(args.output_path)
        colmap_mapper(args.output_path)
        reconstructor(args.output_path)
        visualize(args.output_path, "reconstructed")
    elif args.command == "colmap-dense":
        image_undistorter(args.images_path)
        patch_match_stereo(args.images_path)
        stereo_fusion(args.images_path)
        visualize(args.images_path, "fused")



if __name__ == "__main__":
    main()