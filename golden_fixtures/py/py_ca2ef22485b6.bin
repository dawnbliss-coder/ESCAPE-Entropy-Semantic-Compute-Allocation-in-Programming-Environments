def delete(self, ids):
        """
        Method to delete environments by their id's

        :param ids: Identifiers of environments
        :return: None
        """
        url = build_uri_with_ids('api/v3/environment/%s/', ids)
        return super(ApiEnvironment, self).delete(url)