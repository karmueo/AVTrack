from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.dtb70_path = '/home/tl/work/AVTrack/data/dtb70_path'
    settings.got10k_lmdb_path = '/home/tl/work/AVTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/tl/work/AVTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/tl/work/AVTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/tl/work/AVTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/tl/work/AVTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/tl/work/AVTrack/data/lasot'
    settings.network_path = '/home/tl/work/AVTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/tl/work/AVTrack/data/nfs'
    settings.otb_path = '/home/tl/work/AVTrack/data/otb'
    settings.prj_dir = '/home/tl/work/AVTrack'
    settings.result_plot_path = '/home/tl/work/AVTrack/output/test/result_plots'
    settings.results_path = '/home/tl/work/AVTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/tl/work/AVTrack/output'
    settings.segmentation_path = '/home/tl/work/AVTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/tl/work/AVTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/tl/work/AVTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/tl/work/AVTrack/data/trackingnet'
    settings.uav123_10fps_path = '/home/tl/work/AVTrack/data/uav123_10fps_path'
    settings.uav123_path = '/home/tl/work/AVTrack/data/uav123_path'
    settings.uav_path = '/home/tl/work/AVTrack/data/uav'
    settings.uavdt_path = '/home/tl/work/AVTrack/data/uavdt_path'
    settings.visdrone2018_path = '/home/tl/work/AVTrack/data/visdrone2018_path'
    settings.vot18_path = '/home/tl/work/AVTrack/data/vot2018'
    settings.vot22_path = '/home/tl/work/AVTrack/data/vot2022'
    settings.vot_path = '/home/tl/work/AVTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

