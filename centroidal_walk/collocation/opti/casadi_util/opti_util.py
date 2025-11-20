import rich
from casadi import *
from path_dict import PathDict
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


def describe_variable(self: casadi.Opti, variable: casadi.Opti.variable, var_name, scope: list[str]):
	"""
	Add meta info to opti.variable
	:param self:
	:param variable: may be None
	:param var_name:
	:param scope:
	:return:
	"""
	if variable is None:
		return
	self.update_user_dict(variable, {
		'var_name': var_name,
		'scope': scope
	})



def print_all_opti_vars(opti: Opti):
	# print all opti vars
	print('\n\n')
	# print('Opti variables: ')
	# for var in self.opti.value_variables():
	#    print('>  ', var.is_symbolic())
	#    #print('>  ', self.opti.user_dict(var))
	# test
	opti_ad: casadi.OptiAdvanced = opti.advanced  # makes copy
	# print(opti_ad.x_describe(300))
	# print(opti_ad.active_symvar(casadi.OPTI_PAR))
	rich.print(opti_ad.symvar())

	# sort variables by scopes
	scope_tree = PathDict({})
	for var in opti_ad.symvar():
		# check if it is a variable
		if not hasattr(var, 'name') or not callable(var.name) or not var.is_symbolic():
			print(f'>  var {var} has no name or is not symbolic -> ignore for printing')
			continue

		if not var.name().startswith('opti0_x'):
			continue
		var_dict = opti.user_dict(var)
		if 'scope' in var_dict is not None:
			scope = var_dict['scope']
			add_vars = {
					'__vars': {
						'__var_list': []
					}
				}
			if scope not in scope_tree:
				scope_tree[scope] = add_vars
			elif '__vars' not in scope_tree[scope]:
				scope_tree[scope].update(add_vars)

			var_dict['shape'] = var.shape
			var_dict['numel'] = var.numel()
			#var_dict['stacktrace'] = None
			scope_tree[scope + ['__vars', '__var_list']].append(var_dict)
		else:
			print(f'>  [UNKNOWN_SCOPE] {var.name()} {var.shape} [total {var.numel()}] ', opti.user_dict(var))
	scope_tree_data = scope_tree.data

	def create_scope_data_summarized():
		pass


	def sum_vars_of_scopes(scope: dict, scope_path: list[str] = []):
		scope_num_vars = 0
		for key in list(scope.keys()):
			if key == '__vars':
				total_num_vars = 0
				for var in scope['__vars']['__var_list']:
					num_vars = var["numel"]
					total_num_vars += num_vars
				scope['__vars']['__total_num_vars'] = total_num_vars
				scope_num_vars += total_num_vars

			elif not key.startswith('__'):
				sub_scope_path = scope_path + [key]
				scope_num_vars += sum_vars_of_scopes(scope[key], sub_scope_path)
		scope['__total_num_vars'] = scope_num_vars
		return scope_num_vars

	sum_vars_of_scopes(scope_tree_data)

	# exit(12)
	# rich.print(scope_tree_data)

	# def print_scopes(scope: dict, scope_path: list[str] = []):
	#     for key in scope.keys():
	#         if key == '__vars':
	#             for var in scope['__vars']['__var_list']:
	#                 name = '?'
	#                 if 'var_name' in var is not None:
	#                     name = '-- ' + var['var_name']
	#                 scope_depth = len(scope_path)
	#                 print('\t'*6 + f'{name:<40}  {str(var["shape"]):<10} [total {var["numel"]}]')
	#             print()
	#         elif not key.startswith('__'):
	#             sub_scope_path = scope_path + [key]
	#             #path_str = '.'.join(sub_scope_path) + ':'
	#             path_str = '>'*4*len(scope_path) + (' ' if len(scope_path) > 0 else '') + sub_scope_path[-1] + ':'
	#             total_num_vars = -1
	#             if "__total_num_vars" in scope[key]:
	#                 total_num_vars = scope[key]["__total_num_vars"]
	#             print(f'> {path_str:<50}  [total vars {total_num_vars}]',) # '\t'*(len(sub_scope_path)-1)
	#             print_scopes(scope[key], sub_scope_path)
	# print_scopes(scope_tree_data)

	def fill_tree_scopes(scope: dict, tree: rich.tree.Tree, scope_path: list[str] = []):
		# handle vars
		if '__vars' in scope.keys():
			table = Table()

			table.add_column("var name", justify="right", style="cyan", no_wrap=True)
			table.add_column("shape")
			table.add_column("num vars", justify="right")
			table.add_column("file", justify="right")
			table.show_header = False
			for var in scope['__vars']['__var_list']:
				name = var.get('var_name', '??')
				stacktrace = var.get("stacktrace")
				text_filename = Text(stacktrace['file'].split('/')[-1] + ':' + str(stacktrace['line']), "")
				text_filename.stylize(f"link file://{stacktrace['file']}")
				table.add_row(name, str(var.get('shape')), str(var.get('numel')) + ' total', text_filename)
			# columns = Columns(['a', 'b'], equal=True, expand=True)
			tree.add(table)

		# handle sub scopes
		for key in scope.keys():
			if not key.startswith('__'):
				sub_scope_path = scope_path + [key]
				total_num_vars = scope[key].get("__total_num_vars", -1)
				tree_branch = tree.add(
					f"[bold blue]:arrow_right: {key}  [not bold white] \[total vars [bold red]{total_num_vars}[not bold white]]")
				fill_tree_scopes(scope[key], tree_branch, sub_scope_path)

	var_tree = rich.tree.Tree(f"Opti variables:  \[{scope_tree_data.get('__total_num_vars', -1)}]")
	fill_tree_scopes(scope_tree_data, var_tree)
	rich.print(var_tree)

	print('\n\n')