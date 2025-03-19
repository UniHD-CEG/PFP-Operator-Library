import tvm
from tvm.ir.module import IRModule
from tvm.tir import Schedule
import tvm.meta_schedule as ms

###################### INTERFACE ######################

def apply_schedule(mod: IRModule, schedule_func: callable , function_name: str) -> IRModule:

    schedule = tvm.tir.Schedule(mod)
    tasks = mod.get_global_vars()
    print([t.name_hint for t in tasks])
    for task in tasks:
        name = task.name_hint
        if function_name in name:  # check for "dense", different for "convolution"
            print(name, "applying schedule")
            if "dense" in name or "relu" in name:
                schedule = schedule_func(schedule, name, False, bn=16, factor=4, reorder=True, vectorize=True, unroll=True, parallel=True, blocking=True, packed=False)
            else: # TODO add case for conv
                schedule = schedule_func(schedule, name)

    return schedule.mod


def tune(args, MyModule, target, device_key = None, work_dir = "./tune_tmp", max_trials_global=256, num_trials_per_iter=64):
    # print(MyModule)
    tasks = MyModule.get_global_vars()
    print(f"tune tasks = {tasks}")
    for task in tasks:
        name = task.name_hint
        print(80*'*')
        print(f"** task name = {name} **")
        if name == "main": # FIXME exclude all non prim_funcs, this works for the moment
            continue
        #elif "dense" in name: ## DEBUG, skip dense ## DEBUG, skip dense ## DEBUG, skip dense ## DEBUG, skip dense
        #    continue
        mod_linear = tvm.IRModule.from_expr(MyModule[name].with_attr("global_symbol", "main"))

        has_schedule = False
        if "dense_pfp" in name:  # TODO all operations need this
            if args.tune_dense_custom_schedule:
                print('use dense custom schedule for tuning')
                has_schedule = True
                sch = tvm.tir.Schedule(mod_linear)
                sched_func = lambda _: pfp_dense(sch, stochastic=args.tune_dense_stochastic, func_name="main", name = "".join([i for i in name if not i.isdigit()]),
                        reorder=args.tune_dense_reorder, vectorize=args.tune_dense_vectorize, unroll=args.tune_dense_unroll, parallel=args.tune_dense_parallel, blocking=args.tune_dense_blocking, packed=args.tune_dense_packed)
                print('map dense...')
            else:
                print('use dense MetaScheduler for tuning')
        elif "_pool" in name:        
            if not args.tune_lenet_pool:
                continue  ### required for tuning without pool, tuning PFP LeNet performs better without pool tuning!

        print(f'scheduler tune config: max_trials_global = {max_trials_global}, num_trials_per_iter = {num_trials_per_iter}')
        
        database = ms.tune_tir(
            mod=mod_linear,
            target=str(target) + " -num-cores 64",
            #target=str(target) + " -num-cores 1",
            max_trials_global=max_trials_global,
            num_trials_per_iter=num_trials_per_iter,
            runner=ms.runner.RPCRunner(ms.runner.RPCConfig(tracker_host="localhost", tracker_port=9000, tracker_key=device_key, session_timeout_sec=120)),
            space=ms.space_generator.ScheduleFn(sched_func) if has_schedule else "post-order-apply", 
            work_dir=work_dir,
        )
        sch = ms.tir_integration.compile_tir(database, mod_linear, target)
        if sch is not None:
            new_func = sch.mod["main"].with_attr("global_symbol", name)
            db_mod = tvm.IRModule.from_expr(new_func)
            database.commit_workload(db_mod)
            MyModule.update_func(task, new_func)
        else:
            print("No Schedule found for ", name)
    return MyModule


def load_tuning(MyModule, target, work_dir = "./tune_tmp", print_trace = False):
    database = ms.Database.create("json", work_dir=work_dir, module_equality="structural")
    tasks = MyModule.get_global_vars()
    
    for task in tasks:
        name = task.name_hint
        print(f"task = {name}")
        if name == "main":
            continue

        mod_linear = tvm.IRModule.from_expr(MyModule[name].with_attr("global_symbol", "main"))
        sch = ms.tir_integration.compile_tir(database, mod_linear, target)

        if print_trace:
            print(sch.trace)


        if sch is not None:
            print("Schedule found for:", name)
            new_func = sch.mod["main"].with_attr("global_symbol", name)
            MyModule.update_func(task, new_func)
        else:
            print("No Schedule found for ", name)
    return MyModule

###################### SCHEDULES ######################
"""
Schedules for MM-like operations exist based on:
packed / non-packed: access pattern to the second operand
stochastic/deterministic: stochastic schedules work as templates for the "AutoTuner"-MetaSchedule, deterministic schedules can be apllied directly with manual choosen parameters

Optimized: apply all optimizations

Note: all optimized schedules here can be created out of pfp_dense, current implementation is for readability

"""


def pfp_dense(sch: tvm.tir.Schedule, func_name="main", stochastic=True, bn=16, factor=4, name="dense_pfp", reorder=True, vectorize=True, unroll=True, parallel=True, blocking=False, packed=False):
    """
    Flexible Schedule for a dense PFP operator
    sch: base schedule, created directly from module
    block: if emebedded in Relax, the relevant block must be provided, else just use None
    stochastic: true -> generate stochastic schedule, false -> generate deterministic schedule
    bn: blocking factor, irrelevant if stochatic
    factor: split factor of redution loop, determines unroll factor, irrelevant if stochastic
    name: name of the operation, this schedules works for matrix multicplication like computations (tested: dense/ dense_pfp)
    reorder: whether loops are reordered
    vectorize: vectorize the inner most loop (must be a spatial loop)
    unroll: unroll the second inner most loop
    parallel: apply parallelization to the outer most loop
    blocking: block access to matrix A into blocks of size "bn"
    packed: reorder access to matrix B into packs
    """

    print('FOUND PFP DENSE TUNING SCHEDULE')
    
    if packed:
        name += "_packed"
    try:
        block_dense = sch.get_block(name, func_name=func_name) 
    except:
        name+="_first_layer"
        block_dense = sch.get_block(name, func_name=func_name)

    i, j, k = sch.get_loops(block_dense)
    if blocking:
        if stochastic:
            factors_reduction = sch.sample_partitioned_tile(loop=k, n=2)
            factors_outer = sch.sample_partitioned_tile(loop=i, n=2)
            factors_inner = sch.sample_partitioned_tile(loop=j, n=2)
      
        else:
            factors_reduction = [None, bn]
            factors_outer = [None, bn]
            factors_inner = [None, factor]

        i0, i1 = sch.split(i, factors=factors_reduction)
        j0, j1 = sch.split(j, factors=factors_outer)
        k0, k1 = sch.split(k, factors=factors_inner)
        if reorder:
            sch.reorder(i0, j0, k0, i1, k1, j1)
        if vectorize:
            sch.vectorize(j1)
        if unroll:
            sch.unroll(k1)
        if parallel:
            sch.parallel(i0)
    else:
        if reorder:
            sch.reorder(i, j, k)
        if vectorize:
            sch.vectorize(j)
        if unroll:
            sch.unroll(j)
        if parallel:
            sch.parallel(i)

    if packed:
        if "pfp" in name:  # pfp_dense
            block_A = sch.get_block("packedWM", func_name=func_name)
            bigN, _, littleN = sch.get_loops(block_A)
            sch.vectorize(littleN)
            sch.parallel(bigN)

            block_B = sch.get_block("packedWS", func_name=func_name)
            bigN, _, littleN = sch.get_loops(block_B)
            sch.vectorize(littleN)
            sch.parallel(bigN)
        else:  # dense (regular)
            block_A = sch.get_block("packedW", func_name=func_name)
            bigN, _, littleN = sch.get_loops(block_A)
            sch.vectorize(littleN)
            sch.parallel(bigN)
    return sch

