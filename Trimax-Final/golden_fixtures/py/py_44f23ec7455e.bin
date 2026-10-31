def add_row(self, key: str, default: str=None,
                unit_label: str=None, enable: bool=None):
        """
        Add a single row and re-draw as necessary

        :param key: the name and dict accessor
        :param default: the default value
        :param unit_label: the label that should be \
        applied at the right of the entry
        :param enable: the 'enabled' state (defaults to True)
        :return:
        """
        self.keys.append(ttk.Label(self, text=key))

        self.defaults.append(default)
        self.unit_labels.append(
            ttk.Label(self, text=unit_label if unit_label else '')
        )
        self.enables.append(enable)
        self.values.append(ttk.Entry(self))

        row_offset = 1 if self.title is not None else 0

        for i in range(len(self.keys)):
            self.keys[i].grid_forget()

            self.keys[i].grid(row=row_offset, column=0, sticky='e')
            self.values[i].grid(row=row_offset, column=1)

            if self.unit_labels[i]:
                self.unit_labels[i].grid(row=row_offset, column=3, sticky='w')

            if self.defaults[i]:
                self.values[i].config(state=tk.NORMAL)
                self.values[i].delete(0, tk.END)
                self.values[i].insert(0, self.defaults[i])

            if self.enables[i] in [True, None]:
                self.values[i].config(state=tk.NORMAL)
            elif self.enables[i] is False:
                self.values[i].config(state=tk.DISABLED)

            row_offset += 1

            # strip <Return> and <Tab> bindings, add callbacks to all entries
            self.values[i].unbind('<Return>')
            self.values[i].unbind('<Tab>')

            if self.callback is not None:
                def callback(event):
                    self.callback()

                self.values[i].bind('<Return>', callback)
                self.values[i].bind('<Tab>', callback)