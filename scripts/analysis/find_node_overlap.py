from specialcouscous.utils.slurm import expand_node_range


def find_node_overlap(nodes_list: list[str]) -> list[str]:
    """
    Identify overlapping nodes across multiple HoreKa compute jobs.

    This script helps analyze compute jobs run on the HoreKa supercomputer by identifying overlaps in the lists of compute
    nodes used. Each compute job's output file includes a summary of the nodes utilized at the end of the run. By comparing
    these node lists across different jobs, this script identifies which nodes were consistently involved.

    This can be particularly useful for troubleshooting failed jobs, as overlapping nodes in multiple failed jobs may
    indicate problematic compute nodes. Such nodes can then be excluded in future jobs using the "SBATCH --exclude" option.

    Additionally, comparing the node overlap between failed and successful jobs can help refine the list, removing
    functional nodes that were used in successful jobs, thereby narrowing down the potential source of failure.

    Parameters
    ----------
    nodes_list : list[str]
        List of compute node lists to compare. Format must be: "hkn[0169,0171,0201-0203"

    Returns
    -------
    list[str]
        A list of nodes used in all jobs.
    """
    expanded_nodes_list = [
        set(expand_node_range(nodes)) for nodes in nodes_list
    ]  # Expand ranges.
    return sorted(
        set.intersection(*expanded_nodes_list)
    )  # Return the intersection (overlap).


if __name__ == "__main__":
    # Input strings
    nodes_list = [
        # Enter node lists of compute jobs to identify their node overlap (and thus potentially broken nodes).
        # Example:
        "hkn[0169,0171,0201-0203,0206,0209,0218-0219,0222,0249-0251,0253,0255-0257,0259,0261,0265,0267-0268,0304,0330]",
        "hkn[0257,0259]",
    ]
    overlap = find_node_overlap(nodes_list)
    print("Overlapping nodes:", overlap)
